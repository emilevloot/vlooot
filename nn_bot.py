"""
A computer player that uses the trained value network (2-player games).

    python arena.py nn:v1 greedy          # network v1 against the greedy player

How it plays a turn:
  1. It lists whole turns: every Viking placement (with/without shields)
     x every longship choice (none, or one of the ocean's) x claiming the
     best trophy or not. For its best few placements it also tries the
     second-Viking shield. Tiles go on the fjord the greedy way.
  2. Each turn is played on a copy of the game up to the end of the turn.
  3. The network judges all those end positions at once ("a batch"); a
     finished game is judged by its real result.
  4. It plays the turn with the best predicted final margin.
     With depth 2 ('nn:<model>+2') it first checks its best 3 turns against
     the opponent's best answer to each (judged by the same network).

In its simulations it shuffles the longship bag, so taking a longship
doesn't reveal which one comes out of the bag next.

The network runs on plain numpy here (no PyTorch needed), with the weights
from models/<name>.npz written by train_nn.py.
"""

import copy
import os
import random

import numpy as np

import looot as L
import nn_encode as E

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MARGIN_SCALE = 20.0
_CACHE = {}


# ---------------------------------------------------------------------------
# The network in numpy (the same sums as nn_model.py)
# ---------------------------------------------------------------------------

def check_version(p, path, enc=E):
    """Refuse networks made for another encoding or network layout."""
    if (p["land.c1.w"].shape[1] != enc.LAND_F or p["fjord.c1.w"].shape[1] != enc.FJORD_F
            or p["glob.weight"].shape[1] != enc.GLOB_F or "land.blocks.0.conv.w" not in p):
        raise ValueError("%s was made for an older encoding or network (nn_encode.py / "
                         "nn_model.py have changed since); train it again."
                         % os.path.basename(path))


def n_blocks(p, pre):
    k = 0
    while "%s.blocks.%d.conv.w" % (pre, k) in p:
        k += 1
    return k


class NumpyNet:
    """The straightforward version, kept to test FastNet against."""

    def __init__(self, path):
        d = np.load(path)
        self.p = {k: d[k].astype(np.float32) if d[k].dtype != np.int64 else d[k]
                  for k in d.files}
        check_version(self.p, path)

    def _hexconv(self, x, name, nb):
        w, b = self.p[name + ".w"], self.p[name + ".b"]
        B, N, C = x.shape
        xp = np.concatenate([x, np.zeros((B, 1, C), np.float32)], axis=1)
        nbx = xp[:, nb].reshape(B, N, 6 * C)          # the 6 neighbours side by side
        return x @ w[0] + nbx @ w[1:].reshape(6 * C, -1) + b

    def _board(self, x, pre, nb):
        relu = lambda v: np.maximum(v, 0)
        h = relu(self._hexconv(x, pre + ".c1", nb))
        for k in range(n_blocks(self.p, pre)):
            q = "%s.blocks.%d" % (pre, k)
            whole = np.concatenate([h.mean(1), h.max(1)], axis=1)
            g = whole @ self.p[q + ".pool.weight"].T + self.p[q + ".pool.bias"]
            h = relu(h + self._hexconv(h, q + ".conv", nb) + g[:, None, :])
        h = relu(self._hexconv(h, pre + ".cout", nb))
        h = h.reshape(len(h), -1)
        return relu(h @ self.p[pre + ".fc.weight"].T + self.p[pre + ".fc.bias"])

    def __call__(self, land, fjord, glob):
        """Batch of encoded positions -> (margin in points, win chance)."""
        p = self.p
        relu = lambda v: np.maximum(v, 0)
        land = (land.astype(np.float32) - p["land_mu"]) / p["land_sd"]
        fjord = (fjord.astype(np.float32) - p["fjord_mu"]) / p["fjord_sd"]
        glob = (glob.astype(np.float32) - p["glob_mu"]) / p["glob_sd"]
        a = self._board(land, "land", E.LAND_NB)
        b = self._board(fjord[:, 0], "fjord", E.FJORD_NB)
        c = self._board(fjord[:, 1], "fjord", E.FJORD_NB)
        d = relu(glob @ p["glob.weight"].T + p["glob.bias"])
        h = relu(np.concatenate([a, b, c, d], axis=1) @ p["fc1.weight"].T + p["fc1.bias"])
        h = relu(h @ p["fc2.weight"].T + p["fc2.bias"])
        margin = (h @ p["margin.weight"].T + p["margin.bias"])[:, 0] * MARGIN_SCALE
        win = 1 / (1 + np.exp(-(h @ p["win.weight"].T + p["win.bias"])[:, 0]))
        return margin, win


class FastNet:
    """The network the players use: the same answers as NumpyNet, about
    4x faster. The weights are put in the shape the sums need once, when
    loading. And in a batch of candidate turns the opponent's fjord is the
    same everywhere, and the landscape is the same for all endings of one
    Viking placement: each board part of the network runs only once per
    different input, and results are kept in a cache for later batches."""

    def __init__(self, path, enc=E):
        d = np.load(path)
        p = {k: d[k] for k in d.files}
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        self.E = enc                   # the encoding: nn_encode, or mp_encode (2-4 players)
        check_version(p, path, enc)

        def conv(name):
            w = f32(p[name + ".w"])
            return w[0], f32(w[1:].reshape(6 * w.shape[1], -1)), f32(p[name + ".b"])

        self.boards = {}
        boards = [("land", enc.LAND_NB), ("fjord", enc.FJORD_NB)]
        if "pol_land.c1.w" in p:                   # the proposals' own layers
            boards += [("pol_land", enc.LAND_NB), ("pol_fjord", enc.FJORD_NB)]
        for pre, nb in boards:
            blocks = []
            for k in range(n_blocks(p, pre)):
                q = "%s.blocks.%d" % (pre, k)
                blocks.append((conv(q + ".conv"), f32(p[q + ".pool.weight"].T),
                               f32(p[q + ".pool.bias"])))
            self.boards[pre] = (np.asarray(nb), conv(pre + ".c1"), blocks, conv(pre + ".cout"),
                                f32(p[pre + ".fc.weight"].T), f32(p[pre + ".fc.bias"]))
        # input scaling folded into one multiply-add: (x - mu) / sd = x * a + b
        self.scale = {}
        for name in ("land", "fjord", "glob"):
            mu, sd = f32(p[name + "_mu"]), f32(p[name + "_sd"])
            self.scale[name] = (f32(1 / sd), f32(-mu / sd))
        self.has_proposals = "pol_land.c1.w" in p     # the three-part network's proposals
        self.lin = {}                  # every plain layer: name -> (weight.T, bias)
        for k in p:
            if k.endswith(".weight") and not k.startswith(("land.", "fjord.", "pol_land.", "pol_fjord.")):
                name = k[:-len(".weight")]
                self.lin[name] = (f32(p[k].T), f32(p[name + ".bias"]))
        self.p = p
        self.cache = {}

    def _board(self, x, pre):
        """(the board summed up, every space's own numbers)"""
        nb, c1, blocks, cout, fc_w, fc_b = self.boards[pre]
        a, b = self.scale[pre.replace("pol_", "")]
        h = x.astype(np.float32) * a + b
        B, N, _ = h.shape
        # multi-player encoding: spaces of boards not in this game stay 0
        # and don't count in the averages (see nn_model.Block)
        mask = None
        if pre.endswith("land") and hasattr(self.E, "LF_INPLAY"):
            mask = (x[:, :, self.E.LF_INPLAY] > 0).astype(np.float32)[:, :, None]
            h = h * mask

        def hexconv(h, layer):
            w0, wnb, bias = layer
            C = h.shape[2]
            hp = np.empty((B, N + 1, C), dtype=np.float32)
            hp[:, :N] = h
            hp[:, N] = 0                                   # "no neighbour"
            return h @ w0 + hp[:, nb].reshape(B, N, 6 * C) @ wnb + bias

        h = np.maximum(hexconv(h, c1), 0)
        if mask is not None:
            h = h * mask
        for layer, pool_w, pool_b in blocks:
            mean = h.mean(1) if mask is None else (h * mask).sum(1) / np.maximum(mask.sum(1), 1)
            g = np.concatenate([mean, h.max(1)], axis=1) @ pool_w + pool_b
            h = np.maximum(h + hexconv(h, layer) + g[:, None, :], 0)
            if mask is not None:
                h = h * mask
        cells = np.maximum(hexconv(h, cout), 0)
        if mask is not None:
            cells = cells * mask
        return np.maximum(cells.reshape(B, -1) @ fc_w + fc_b, 0), cells

    CACHE_SIZE = 20000

    def _board_unique(self, x, pre):
        """_board on only the inputs it hasn't seen yet. Results are kept in
        a cache, so a board that comes back (the opponent's fjord all turn,
        the same landscape for every ending) is worked out only once."""
        cache = self.cache.setdefault(pre, {})
        if len(cache) + len(x) > self.CACHE_SIZE:
            cache.clear()            # full: start again (before looking anything up)
        keys = [r.tobytes() for r in x]
        todo = {}
        for i, k in enumerate(keys):
            if k not in cache and k not in todo:
                todo[k] = i
        if todo:
            out, cells = self._board(x[list(todo.values())], pre)
            for k, v, c in zip(todo, out, cells):
                cache[k] = (v, c)
        return (np.stack([cache[k][0] for k in keys]),
                np.stack([cache[k][1] for k in keys]))

    def _layer(self, x, name, relu=True):
        w, b = self.lin[name]
        y = x @ w + b
        return np.maximum(y, 0) if relu else y

    def explain(self, land, fjord, glob):
        """(margin in points, win chance, the messages between the parts)"""
        a, cells = self._board_unique(land, "land")
        bs, fcs = zip(*[self._board_unique(fjord[:, k], "fjord") for k in range(fjord.shape[1])])
        b, c, fb, fc = self._seats(np.stack(bs, 1), np.stack(fcs, 1), glob)
        ga, gb = self.scale["glob"]
        g = glob.astype(np.float32) * ga + gb
        h, told = self._head(land, glob, g, a, cells, b, c, (fb, fc))
        h = self._layer(self._layer(h, "fc1"), "fc2")
        margin = self._layer(h, "margin", False)[:, 0] * MARGIN_SCALE
        win = 1 / (1 + np.exp(-self._layer(h, "win", False)[:, 0]))
        return margin, win, told

    def __call__(self, land, fjord, glob):
        """Batch of encoded positions -> (margin in points, win chance)."""
        return self.explain(land, fjord, glob)[:2]

    def _seats(self, bs, fcs, glob):
        """See nn_model.ValueNet.seats: my fjord, and the opponents' (the
        largest number over the opponents in the game)."""
        if bs.shape[1] == 2:
            return bs[:, 0], bs[:, 1], fcs[:, 0], fcs[:, 1]
        on = glob[:, self.E.PRESENT_COLS] > 0
        c = np.where(on[:, :, None], bs[:, 1:], -1e9).max(1)
        c = np.where(on.any(1)[:, None], c, 0).astype(np.float32)
        fc = np.where(on[:, :, None, None], fcs[:, 1:], -1e9).max(1)
        fc = np.where(on.any(1)[:, None, None], fc, 0).astype(np.float32)
        return bs[:, 0], c, fcs[:, 0], fc

    def _head(self, land, glob, g, a, cells, b, c, fj_cells):
        """The input of fc1 (see nn_model.ValueNet.head)."""
        return np.concatenate([a, b, c, self._layer(g, "glob")], axis=1), {}


def _masked_max(x, mask):
    m = np.where(mask, x, -1e9).max(1)
    return np.where(mask.any(1), m, 0).astype(np.float32)


def _masked_mean(x, mask):
    mf = mask.astype(np.float32)
    return (x * mf).sum(1) / np.maximum(mf.sum(1), 1)


class ThreeNet(FastNet):
    """The three-part network (nn_model.ThreePartNet) in numpy."""

    def _mlp(self, x, name):
        return self._layer(self._layer(x, name + ".0"), name + ".2", False)

    def _head(self, land, glob, g, a, cells, b, c, fj_cells):
        land = land.astype(np.float32)
        glob = glob.astype(np.float32)
        clock = g[:, self.E.CLOCK_COLS]
        # 1. fjord part: item values for me and for the opponent
        val_me = self._mlp(np.concatenate([b, c, clock], 1), "values")
        val_opp = self._mlp(np.concatenate([c, b, clock[:, ::-1]], 1), "values")
        # 2. board part
        free = land[:, :, self.E.LF_FREE]
        res = land[:, :, self.E.RES_COLS] * free[:, :, None]
        got_me = np.concatenate([res, land[:, :, self.E.GAIN_BLD[0]]], 2)
        got_opp = np.concatenate([res, land[:, :, self.E.GAIN_BLD[1]]], 2)
        place_me = (got_me * val_me[:, None, :]).sum(2)
        place_opp = (got_opp * val_opp[:, None, :]).sum(2)
        score = self._mlp(np.concatenate([cells, place_me[:, :, None], place_opp[:, :, None],
                                          free[:, :, None]], 2), "cell")
        on = free > 0
        pooled = np.stack([_masked_max(place_me, on), _masked_mean(place_me, on),
                           _masked_max(place_opp, on), _masked_mean(place_opp, on),
                           _masked_max(score[:, :, 0], on), _masked_mean(score[:, :, 0], on),
                           _masked_max(score[:, :, 1], on), _masked_mean(score[:, :, 1], on)], 1)
        offer_me = got_me.sum(1) / 10
        offer_opp = got_opp.sum(1) / 10
        # 3. ship part
        need = glob[:, self.E.SHIP_NEED]
        bonus = glob[:, self.E.SHIP_BONUS] * np.asarray(self.E.BONUS_ON, np.float32)
        present = need.sum(2) > 0
        fits = []
        for val, offer, clk in ((val_me, offer_me, clock), (val_opp, offer_opp, clock[:, ::-1])):
            v, o = val[:, None, :], offer[:, None, :]
            x = np.concatenate([need, bonus, need * v[:, :, :4], bonus * v, need * o[:, :, :4],
                                np.broadcast_to(clk[:, None, :], (len(clk), 5, 2))], 2)
            fits.append(self._mlp(x, "ship")[:, :, 0])
        ships = np.stack([_masked_max(fits[0], present), _masked_mean(fits[0], present),
                          _masked_max(fits[1], present), _masked_mean(fits[1], present)], 1)
        d = self._layer(g, "glob")
        msg = np.concatenate([val_me, val_opp, pooled, offer_me, offer_opp, ships], 1)
        told = {"values_me": val_me, "values_opp": val_opp, "place_me": place_me,
                "ships_me": np.where(present, fits[0], 0), "present": present}
        self._inner = (cells, place_me, place_opp, free, land[:, :, self.E.LF_STACK],
                       fj_cells[0], val_me, fits[0])
        return np.concatenate([a, b, c, d, msg], 1), told

    def proposals(self, land, fjord, glob):
        """The proposals (nn_model.ThreePartNet.proposals) for positions at the
        start of a turn: raw scores pol_place [B, 55, 3], pol_tile [B, 37, 8],
        pol_ship [B, 6] (5 ocean slots, then "none")."""
        a, cells = self._board_unique(land, "land")
        bs, fcs = zip(*[self._board_unique(fjord[:, k], "fjord") for k in range(fjord.shape[1])])
        b, c, fb, fc = self._seats(np.stack(bs, 1), np.stack(fcs, 1), glob)
        ga, gb = self.scale["glob"]
        g = glob.astype(np.float32) * ga + gb
        hin, _ = self._head(land, glob, g, a, cells, b, c, (fb, fc))
        h = self._layer(self._layer(hin, "fc1"), "fc2")
        cells, place_me, place_opp, free, stack, fb, val_me, fits = self._inner
        own_l = self._board(land, "pol_land")[1]
        own_f = self._board(fjord[:, 0], "pol_fjord")[1]
        place = self._mlp(np.concatenate([own_l, cells, place_me[:, :, None], place_opp[:, :, None],
                                          free[:, :, None], stack.astype(np.float32)[:, :, None]], 2),
                          "pol_place")
        tile = self._mlp(np.concatenate([own_f, fb, np.broadcast_to(
            val_me[:, None, :], (len(val_me), fb.shape[1], val_me.shape[1]))], 2), "pol_tile")
        ship = np.concatenate([fits, self._layer(h, "pol_none", False)], 1)
        return {"pol_place": place, "pol_tile": tile, "pol_ship": ship}


class AttnNet(FastNet):
    """The network with attention (nn_model.AttnNet) in numpy."""

    def _ln(self, x, name):
        w, b = self.p[name + ".weight"].astype(np.float32), self.p[name + ".bias"].astype(np.float32)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * w + b

    def _head(self, land, glob, g, a, cells, b, c, fj_cells):
        lin = lambda x, name: self._layer(x, name, False)
        glob = glob.astype(np.float32)
        need = glob[:, self.E.SHIP_NEED]
        bonus = glob[:, self.E.SHIP_BONUS] * np.asarray(self.E.BONUS_ON, np.float32)
        fme, fopp = fj_cells
        x = np.concatenate([lin(cells, "tok_land"), lin(fme, "tok_fjord"), lin(fopp, "tok_fjord"),
                            lin(np.concatenate([need, bonus], 2) / 3, "tok_ship"),
                            lin(g, "tok_glob")[:, None, :]], 1) + self.p["pos"].astype(np.float32)
        B, T, D = x.shape
        H = 2                                  # heads (nn_model.AttnNet.HEADS)
        k = 0
        while "layers.%d.qkv.weight" % k in self.p:
            q_ = "layers.%d" % k
            qkv = lin(self._ln(x, q_ + ".ln1"), q_ + ".qkv").reshape(B, T, 3, H, D // H)
            q, kk, v = qkv.transpose(2, 0, 3, 1, 4)                  # [B, H, T, D/H] each
            att = q @ kk.transpose(0, 1, 3, 2) / np.float32(np.sqrt(D // H))
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            x = x + lin((att @ v).transpose(0, 2, 1, 3).reshape(B, T, D), q_ + ".out")
            x = x + lin(np.maximum(lin(self._ln(x, q_ + ".ln2"), q_ + ".ff1"), 0), q_ + ".ff2")
            k += 1
        x = self._ln(x, "ln")
        d = self._layer(g, "glob")
        return np.concatenate([a, b, c, d, x[:, -1], x.mean(1)], 1), {}


def load_net(name):
    """The network models/<name>.npz, as the right kind (see nn_model.ARCHS)."""
    path = name if name.endswith(".npz") else os.path.join(MODEL_DIR, name + ".npz")
    if path not in _CACHE:
        with np.load(path) as d:
            arch = str(d["arch"]) if "arch" in d.files else "value"
        enc = E
        if arch.endswith("4"):                     # the 2-4-player networks
            import mp_encode
            enc, arch = mp_encode, arch[:-1]
        _CACHE[path] = {"value": FastNet, "three": ThreeNet, "attn": AttnNet}[arch](path, enc)
    return _CACHE[path]


def thoughts(net, g, me):
    """What the network thinks of game g for player `me`, for the web page.
    Everything is ASKED of the network, in points of final margin:
      values       one extra item of each kind on my fjord now (where the
                   player would put it): how many points better I stand
      values_opp   the same for the opponent, from the opponent's side
      ships        taking each longship in the ocean now: points better/worse
      best_spaces  the free spaces where a Viking of mine is worth most
    (The three-part network's own messages stay inside it: their numbers are
    signals it learned to use, not points, so they are not shown.)"""
    helper = L.Bot(random.Random(0))

    def margins(games, who):
        enc = [net.E.encode(x, who) for x in games]
        m, w = net(np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]),
                   np.stack([e[2] for e in enc]))
        return m, w

    def item_values(who):
        games, names = [g], []
        for t in E.SITE_ITEMS:
            g2 = copy.deepcopy(g)
            p = g2.players[who]
            c = helper.best_tile_cell(g2, p, {"type": t})
            if c is None:                      # fjord full
                continue
            p.fjord[c] = {"kind": "res" if t in E.RES else "bld", "type": t}
            p.update_fills()
            games.append(g2)
            names.append(t)
        m, w = margins(games, who)
        return m, w, {t: round(float(v - m[0]), 1) for t, v in zip(names, m[1:])}

    m, w, values = item_values(me)
    # the opponent to show: with more players the strongest one (highest score)
    rival = 1 - me if len(g.players) == 2 else net.E.strongest(g, me)
    out = {"player": me, "margin": round(float(m[0]), 1), "win": round(float(w[0]), 3),
           "values": values, "values_opp": item_values(rival)[2]}

    # longships: as if I took each one now
    games, slots = [], []
    for slot, sid in enumerate(g.ocean_ships):
        if sid is None:
            continue
        g2 = copy.deepcopy(g)
        p = g2.players[me]
        need, cat, bonus = L.LONGSHIPS[sid]
        c = helper.best_ship_cell(g2, p, list(need))
        if c is None:
            continue
        p.fjord[c] = {"kind": "ship", "id": sid, "need": list(need), "cat": cat,
                      "bonus": bonus, "filled": False}
        g2.ocean_ships[slot] = None
        p.update_fills()
        games.append(g2)
        slots.append(slot)
    out["ships"] = [None] * len(g.ocean_ships)
    if games:
        for slot, v in zip(slots, margins(games, me)[0]):
            out["ships"][slot] = round(float(v - m[0]), 1)

    # Viking placements: as if it were my turn, turn ended afterwards
    out["best_spaces"] = []
    if not g.game_over and g.players[me].vikings_left > 0:
        g2 = copy.deepcopy(g)
        g2.current, g2.phase, g2.pending = me, "place", []
        g2.placed_this_turn, g2.took_ship, g2.extra_active = 0, False, False
        games, cells = [], []
        for c in g2.legal_cells(False):
            g3 = copy.deepcopy(g2)
            g3.place_viking(c)
            helper.place_pending(g3)
            g3.end_turn()
            if not g3.game_over:
                games.append(g3)
                cells.append(c)
        if games:
            v = margins(games, me)[0]
            out["best_spaces"] = [L.key(cells[i]) for i in np.argsort(-v)[:3]]
    return out


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------

class NNBot(L.Bot):
    TOP_FOR_EXTRA = 3       # try the 2nd-Viking shield after the best few placements

    def __init__(self, rng, net_name, explore=0.0, ship_cost=0.0, depth=1, endgame=None):
        super().__init__(rng)
        self.net_name = net_name
        self.net = load_net(net_name)
        # "wide" or "narrow": the last 2 turns worked out exactly (endgame.py)
        # instead of judged by the network; None: the network all game
        self.endgame = endgame
        self.enc = self.net.E          # its encoding (nn_encode, or mp_encode for 2-4 players)
        self.explore = explore
        self.depth = depth
        # Only take a longship if the network thinks it is at least this many
        # points better: out of several noisy estimates, the highest is
        # usually too high ("optimizer's curse"), and a wrong longship costs 5.
        self.ship_cost = ship_cost
        # with the network's proposals, they pick the fjord spaces to try and
        # add placements to the shortlist (see fastgame.POL_K)
        self.use_policy = getattr(self.net, "has_proposals", False)
        self.proposer = self.net
        self.pmap = None

    POL_K, POL_SHIP_K, GREEDY_HELP = 2, 2, 1
    TOP_POL = 2

    def _proposals(self, g):
        """The proposals for the player to move: (place [NL, 3], tile [NF, 8])."""
        land, fjord, glob = self.enc.encode(g, g.current)
        m = self.proposer.proposals(land[None], fjord[None], glob[None])
        return np.asarray(m["pol_place"][0], np.float64), np.asarray(m["pol_tile"][0], np.float64)

    def _merge(self, p, col, kp, greedy):
        """The kp best empty spaces by the waar-kaart column col, then the
        greedy spaces not among them: [(proposal score, space)]."""
        tile = self.pmap[1]
        fi = self.enc.FJORD_INDEX
        empty = [c for c in p.empty_cells()]
        order = sorted(range(len(empty)), key=lambda i: (-tile[fi[empty[i]], col], i))
        out = [(tile[fi[empty[i]], col], empty[i]) for i in order[:kp]]
        have = [c for _, c in out]
        for c in greedy:
            if c not in have:
                out.append((tile[fi[c], col], c))
                have.append(c)
        return out

    def _tile_choices(self, g, p, item_type):
        """[(score, space)] to try for a tile: greedy's best TILE_K, or with
        the proposals: their best POL_K plus greedy's best GREEDY_HELP."""
        if self.pmap is None:
            return self._ranked_tile_cells(g, p, item_type, self.TILE_K)
        greedy = [c for _, c in self._ranked_tile_cells(g, p, item_type, self.GREEDY_HELP)]
        return self._merge(p, self.enc.SITE_ITEMS.index(item_type), self.POL_K, greedy)

    def _ship_choices(self, g, p, need):
        if self.pmap is None:
            return self._ranked_ship_cells(g, p, need, self.SHIP_K)
        greedy = self._ranked_ship_cells(g, p, need, self.GREEDY_HELP)
        return [c for _, c in self._merge(p, 7, self.POL_SHIP_K, greedy)]

    # The network chooses the fjord layout too: every tile tries its TILE_K
    # best spaces by the greedy rule, every longship its SHIP_K best spaces,
    # and the network judges where each way ends.
    TILE_K = 3              # spaces tried per tile
    MAX_LAYOUTS = 4         # fjord layouts kept per Viking placement
    SHIP_K = 2              # spaces tried per longship

    def _ranked_tile_cells(self, g, p, item_type, k):
        """The k best empty fjord spaces for a tile by the greedy rule
        (looot.Bot.best_tile_cell), best first: [(score, space)]."""
        out = []
        for idx, c in enumerate(p.empty_cells()):
            v = 0.0
            for n in L.neighbours(c):
                it = p.fjord.get(n)
                if it and it["kind"] in ("ship", "site") and                         not it.get("filled") and not it.get("done"):
                    missing = list(it["need"])
                    for h in p.adjacent_items(n):
                        if h in missing:
                            missing.remove(h)
                    if item_type in missing:
                        v += self.w["tile_need"]
                    else:
                        v -= self.w["tile_block"]
                elif n in p.fjord_cells and n not in p.fjord:
                    v += self.w["tile_open"]
            v += self.rng.random() * 0.1
            out.append((v, idx, c))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [(v, c) for v, _, c in out[:k]]

    def _ranked_ship_cells(self, g, p, need, k):
        """The k best empty fjord spaces for a longship (looot.Bot.best_ship_cell)."""
        out = []
        for idx, c in enumerate(p.empty_cells()):
            m = L.matched_count(need, p.adjacent_items(c))
            room = sum(1 for n in L.neighbours(c) if n in p.fjord_cells and n not in p.fjord)
            v = (m * self.w["shipcell_match"] + min(room, 3 - m) * self.w["shipcell_room"] +
                 self.rng.random() * 0.1)
            if room < 3 - m:
                v -= 20
            out.append((v, idx, c))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [c for v, _, c in out[:k]]

    # --- trying things on copies, remembering the exact moves ---
    def _layouts(self, g, opt):
        """Copy g, place a Viking, and put its tiles on the fjord in up to
        MAX_LAYOUTS ways: every tile tries its TILE_K best spaces; of all
        combinations, those the greedy rule scores highest are kept.
        Returns [(copy, [tile moves])] so the moves can be replayed."""
        g2 = copy.deepcopy(g)
        g2.place_viking(*opt)
        layouts = [(0.0, g2, [])]
        while layouts[0][1].phase == "tiles":
            nxt = []
            for score, gl, tiles in layouts:
                p = gl.player()
                i = 0
                if len(gl.pending) > len(p.empty_cells()):
                    i = max(range(len(gl.pending)),
                            key=lambda j: self.KEEP_ORDER[gl.pending[j]["type"]])
                for v, c in self._tile_choices(gl, p, gl.pending[i]["type"]):
                    g3 = copy.deepcopy(gl)
                    g3.place_tile(i, c)
                    nxt.append((score + v, g3, tiles + [(i, c)]))
            nxt.sort(key=lambda x: -x[0])      # (stable: equal scores keep their order)
            layouts = nxt[:self.MAX_LAYOUTS]
        return [(gl, tiles) for _, gl, tiles in layouts]

    def _endings(self, g2):
        """Every way to finish the turn from g2: longship (none or one) x
        trophy (none or the best one). Yields (plan, finished copy)."""
        p = g2.player()
        ships = [None]
        if not g2.took_ship and p.empty_cells():
            ships += [s for s, sid in enumerate(g2.ocean_ships) if sid is not None]
        for slot in ships:
            ways = [(None, copy.deepcopy(g2))]
            if slot is not None:
                g3 = copy.deepcopy(g2)
                # The copy's bag is in the real order, so taking a longship
                # would show which one is drawn next - something a player can't
                # know. Shuffling the copy's bag makes it a random draw.
                self.rng.shuffle(g3.bag)
                g3.pick_ship(slot)
                need = list(L.LONGSHIPS[g3.ocean_ships[slot]][0])
                ways = []
                for cell in self._ship_choices(g3, g3.player(), need):
                    g4 = copy.deepcopy(g3)
                    g4.place_ship(cell)
                    ways.append((cell, g4))
            for cell, g3 in ways:
                trophies = [None]
                cl = g3.claimable_trophies(g3.player())
                if cl:
                    trophies.append(max(cl))
                for tr in trophies:
                    g4 = copy.deepcopy(g3) if len(trophies) > 1 else g3
                    if tr is not None:
                        g4.claim_trophy(tr)
                    g4.end_turn()
                    yield (slot, cell, tr), g4

    def _judge(self, games, me):
        """Predicted final margin for `me` in each finished-turn copy."""
        vals = np.zeros(len(games))
        todo = []
        for k, g in enumerate(games):
            if g.game_over:
                vals[k] = self.enc.outcome(g, me)[0]   # known exactly
            else:
                todo.append(k)
        if todo:
            enc = [self.enc.encode(games[k], me) for k in todo]
            margin, _ = self.net(np.stack([e[0] for e in enc]),
                                 np.stack([e[1] for e in enc]),
                                 np.stack([e[2] for e in enc]))
            vals[todo] = margin
        return vals

    TOP_FULL = 4            # placements worked out with every ending

    def _plain_end(self, g2):
        """g2 finished without longship or trophy: a quick first look."""
        g3 = copy.deepcopy(g2)
        g3.end_turn()
        return g3

    def _shortlist(self, placed, me, k):
        """Judge placements quickly (no longship/trophy) and keep the best k.
        placed: list of (opt, copy after placing, tile moves, ...)."""
        if len(placed) <= k:
            return placed
        vals = self._judge([self._plain_end(p[1]) for p in placed], me)
        order = np.argsort(-vals)[:k]
        return [placed[i] for i in order]

    DEPTH2_TOP = 3          # with depth 2: moves checked against the best reply

    def play_turn(self, g):
        me = g.current
        if (self.endgame and not self.explore and g.phase == "place"
                and 1 <= g.players[me].vikings_left <= 2):
            import endgame
            r = endgame.reach(g, self.net_name,
                              endgame.WIDE if self.endgame == "wide" else endgame.NARROW,
                              tie_break=True)
            if r is not None:
                self._apply(g, r[1])
                return
        cands, vals = self._candidates(g)
        if not cands:
            if g.phase == "actions":
                g.end_turn()
            return
        # choose (sometimes at random, when exploring for training data)
        if self.explore and self.rng.random() < self.explore:
            k = self.rng.randrange(len(cands))
        elif self.depth >= 2:
            k = self._look_ahead(cands, vals, me)
        else:
            k = int(np.argmax(vals + np.array([self.rng.random() * 1e-3 for _ in vals])))
        self._apply(g, cands[k])

    def _look_ahead(self, cands, vals, me):
        """Depth 2: take the best few moves, let the opponent answer each with
        ITS best move (judged by the same network from its side), and pick
        the move that is best for me after that answer."""
        top = list(np.argsort(-vals)[:self.DEPTH2_TOP])
        after = []
        for k in top:
            g4 = cands[k][5]
            if g4.game_over or g4.current == me:
                after.append(g4)           # no answer to wait for
                continue
            g5 = copy.deepcopy(g4)
            oc, ov = self._candidates(g5)
            after.append(oc[int(np.argmax(ov))][5] if oc else g5)
        v2 = self._judge(after, me)
        return top[int(np.argmax(v2))]

    def _apply(self, g, cand):
        """Replay a chosen turn on the real game."""
        opt, tiles, opt2, tiles2, (slot, cell, tr), _ = cand
        if opt is not None:
            g.place_viking(*opt)
            for i, c in tiles:
                g.place_tile(i, c)
        if opt2 is not None:
            g.use_extra_shield()
            g.place_viking(*opt2)
            for i, c in tiles2:
                g.place_tile(i, c)
        if slot is not None:
            g.pick_ship(slot)
            g.place_ship(cell)
        if tr is not None:
            g.claim_trophy(tr)
        g.end_turn()

    def _candidates(self, g):
        """All whole turns for the player to move in g, each with the
        network's judgement of where it ends. Returns (cands, vals)."""
        me = g.current
        self.pmap = self._proposals(g) if self.use_policy else None
        # Stage 1: every placement, judged quickly; keep the best few
        if g.phase == "place":
            opts = self.placement_options(g)
            firsts, row_opt = [], []
            for o, opt in enumerate(opts):
                for g2, tiles in self._layouts(g, opt):
                    firsts.append((opt, g2, tiles))
                    row_opt.append(o)
            if len(firsts) > self.TOP_FULL:
                vals = self._judge([self._plain_end(f[1]) for f in firsts], me)
                idx = list(np.argsort(-vals)[:self.TOP_FULL])
                if self.pmap is not None:
                    # also the placements the proposal likes best (their best layout)
                    li = self.enc.LAND_INDEX
                    score = np.array([self.pmap[0][li[c], 2 if occ else (1 if dbl else 0)]
                                      for c, occ, dbl in opts])
                    row_opt = np.array(row_opt)
                    for o in np.argsort(-score, kind="stable")[:self.TOP_POL]:
                        rows = np.where(row_opt == o)[0]
                        r = int(rows[np.argmax(vals[rows])])
                        if r not in idx:
                            idx.append(r)
                firsts = [firsts[i] for i in idx]
        else:
            firsts = [(None, copy.deepcopy(g), [])]
        # each candidate: (placement, tiles, extra placement, extra tiles, ending plan, end state)
        cands = []
        for opt, g2, tiles in firsts:
            # Stage 2: the shortlisted placements with every ending
            for plan, g4 in self._endings(g2):
                cands.append((opt, tiles, None, None, plan, g4))
            # and with a 2nd Viking (shield), again shortlisted first
            if opt is not None and g2.can_use_extra(g2.player()):
                g2x = copy.deepcopy(g2)
                g2x.use_extra_shield()
                seconds = [(o2, g3, t2) for o2 in self.placement_options(g2x)
                           for g3, t2 in self._layouts(g2x, o2)]
                for opt2, g3, tiles2 in self._shortlist(seconds, me, 2):
                    for plan, g4 in self._endings(g3):
                        cands.append((opt, tiles, opt2, tiles2, plan, g4))
        if not cands:
            return [], np.zeros(0)
        vals = self._judge([c[5] for c in cands], me)
        if self.ship_cost:
            vals = vals - np.array([self.ship_cost if c[4][0] is not None else 0.0
                                    for c in cands])
        return cands, vals


def player_from_name(name, rng, explore=0.0):
    """'nn:v1' (or 'v1') -> NNBot using models/v1.npz.
    'nn:v1@3'  -> takes a longship only if it looks 3 points better.
    'nn:v1+2'  -> looks 2 turns deep (its move and the opponent's answer).
    'nn:v1=g'  -> the fjord laid out by the greedy rule (as before the
                  network chose the layout itself).
    'nn:v1!e'  -> its last 2 turns worked out exactly (endgame.py).
    Can be combined: 'nn:v1@2+2', 'nn:v1+2=g', 'nn:v1!e'."""
    model = name.split(":", 1)[-1]
    endgame = None
    if model.endswith("!e"):
        model, endgame = model[:-2], "wide"
    greedy_fjord = model.endswith("=g")
    if greedy_fjord:
        model = model[:-2]
    depth = 1
    if "+" in model:
        model, d = model.split("+")
        depth = int(d)
    ship_cost = 0.0
    if "@" in model:
        model, cost = model.split("@")
        ship_cost = float(cost)
    bot = NNBot(rng, model, explore, ship_cost, depth, endgame)
    if greedy_fjord:                   # the old way: tiles and longships where greedy puts them
        bot.TILE_K = bot.MAX_LAYOUTS = bot.SHIP_K = 1
        bot.use_policy = False
    return bot
