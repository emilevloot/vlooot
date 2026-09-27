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

class NumpyNet:
    """The straightforward version, kept to test FastNet against."""

    def __init__(self, path):
        d = np.load(path)
        self.p = {k: d[k].astype(np.float32) if d[k].dtype != np.int64 else d[k]
                  for k in d.files}
        if (self.p["land.c1.w"].shape[1] != E.LAND_F or self.p["fjord.c1.w"].shape[1] != E.FJORD_F
                or self.p["glob.weight"].shape[1] != E.GLOB_F):
            raise ValueError("%s was trained for an older encoding (nn_encode.py "
                             "has changed since); train it again." % os.path.basename(path))

    def _hexconv(self, x, name, nb):
        w, b = self.p[name + ".w"], self.p[name + ".b"]
        B, N, C = x.shape
        xp = np.concatenate([x, np.zeros((B, 1, C), np.float32)], axis=1)
        nbx = xp[:, nb].reshape(B, N, 6 * C)          # the 6 neighbours side by side
        return x @ w[0] + nbx @ w[1:].reshape(6 * C, -1) + b

    def _board(self, x, pre, nb):
        relu = lambda v: np.maximum(v, 0)
        h = relu(self._hexconv(x, pre + ".c1", nb))
        h = relu(h + self._hexconv(h, pre + ".c2", nb))
        h = relu(h + self._hexconv(h, pre + ".c3", nb))
        h = relu(self._hexconv(h, pre + ".c4", nb))
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

    def __init__(self, path):
        d = np.load(path)
        p = {k: d[k] for k in d.files}
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        if (p["land.c1.w"].shape[1] != E.LAND_F or p["fjord.c1.w"].shape[1] != E.FJORD_F
                or p["glob.weight"].shape[1] != E.GLOB_F):
            raise ValueError("%s was trained for an older encoding; train it again." % path)
        self.boards = {}
        for pre, nb in (("land", E.LAND_NB), ("fjord", E.FJORD_NB)):
            layers = []
            for c in ("c1", "c2", "c3", "c4"):
                w = f32(p["%s.%s.w" % (pre, c)])
                cin = w.shape[1]
                layers.append((w[0], f32(w[1:].reshape(6 * cin, -1)), f32(p["%s.%s.b" % (pre, c)])))
            self.boards[pre] = (np.asarray(nb), layers, f32(p[pre + ".fc.weight"].T),
                                f32(p[pre + ".fc.bias"]))
        # input scaling folded into one multiply-add: (x - mu) / sd = x * a + b
        self.scale = {}
        for name in ("land", "fjord", "glob"):
            mu, sd = f32(p[name + "_mu"]), f32(p[name + "_sd"])
            self.scale[name] = (f32(1 / sd), f32(-mu / sd))
        self.glob_w, self.glob_b = f32(p["glob.weight"].T), f32(p["glob.bias"])
        self.fc1_w, self.fc1_b = f32(p["fc1.weight"].T), f32(p["fc1.bias"])
        self.fc2_w, self.fc2_b = f32(p["fc2.weight"].T), f32(p["fc2.bias"])
        self.m_w, self.m_b = f32(p["margin.weight"].T), f32(p["margin.bias"])
        self.w_w, self.w_b = f32(p["win.weight"].T), f32(p["win.bias"])
        self.cache = {}

    def _board(self, x, pre):
        nb, layers, fc_w, fc_b = self.boards[pre]
        a, b = self.scale[pre]
        h = x.astype(np.float32) * a + b
        B, N, _ = h.shape
        for k, (w0, wnb, bias) in enumerate(layers):
            C = h.shape[2]
            hp = np.empty((B, N + 1, C), dtype=np.float32)
            hp[:, :N] = h
            hp[:, N] = 0                                   # "no neighbour"
            out = h @ w0 + hp[:, nb].reshape(B, N, 6 * C) @ wnb + bias
            if k in (1, 2):                                # residual layers
                out += h
            h = np.maximum(out, 0)
        return np.maximum(h.reshape(B, -1) @ fc_w + fc_b, 0)

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
            out = self._board(x[list(todo.values())], pre)
            for k, v in zip(todo, out):
                cache[k] = v
        return np.stack([cache[k] for k in keys])

    def __call__(self, land, fjord, glob):
        """Batch of encoded positions -> (margin in points, win chance)."""
        a = self._board_unique(land, "land")
        b = self._board_unique(fjord[:, 0], "fjord")
        c = self._board_unique(fjord[:, 1], "fjord")
        ga, gb = self.scale["glob"]
        d = np.maximum((glob.astype(np.float32) * ga + gb) @ self.glob_w + self.glob_b, 0)
        h = np.maximum(np.concatenate([a, b, c, d], axis=1) @ self.fc1_w + self.fc1_b, 0)
        h = np.maximum(h @ self.fc2_w + self.fc2_b, 0)
        margin = (h @ self.m_w + self.m_b)[:, 0] * MARGIN_SCALE
        win = 1 / (1 + np.exp(-(h @ self.w_w + self.w_b)[:, 0]))
        return margin, win


def load_net(name):
    path = name if name.endswith(".npz") else os.path.join(MODEL_DIR, name + ".npz")
    if path not in _CACHE:
        _CACHE[path] = FastNet(path)
    return _CACHE[path]


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------

class NNBot(L.Bot):
    TOP_FOR_EXTRA = 3       # try the 2nd-Viking shield after the best few placements

    def __init__(self, rng, net_name, explore=0.0, ship_cost=0.0, depth=1):
        super().__init__(rng)
        self.net = load_net(net_name)
        self.explore = explore
        self.depth = depth
        # Only take a longship if the network thinks it is at least this many
        # points better: out of several noisy estimates, the highest is
        # usually too high ("optimizer's curse"), and a wrong longship costs 5.
        self.ship_cost = ship_cost

    # --- trying things on copies, remembering the exact moves ---
    def _place(self, g, opt):
        """Copy g, place a Viking, put its tiles on the fjord.
        Returns (copy, [tile moves]) so the same moves can be replayed."""
        g2 = copy.deepcopy(g)
        g2.place_viking(*opt)
        tiles = []
        p = g2.player()
        while g2.phase == "tiles":
            i = 0
            if len(g2.pending) > len(p.empty_cells()):
                i = max(range(len(g2.pending)),
                        key=lambda j: self.KEEP_ORDER[g2.pending[j]["type"]])
            c = self.best_tile_cell(g2, p, g2.pending[i])
            g2.place_tile(i, c)
            tiles.append((i, c))
        return g2, tiles

    def _endings(self, g2):
        """Every way to finish the turn from g2: longship (none or one) x
        trophy (none or the best one). Yields (plan, finished copy)."""
        p = g2.player()
        ships = [None]
        if not g2.took_ship and p.empty_cells():
            ships += [s for s, sid in enumerate(g2.ocean_ships) if sid is not None]
        for slot in ships:
            g3 = copy.deepcopy(g2)
            cell = None
            if slot is not None:
                # The copy's bag is in the real order, so taking a longship
                # would show which one is drawn next - something a player can't
                # know. Shuffling the copy's bag makes it a random draw.
                self.rng.shuffle(g3.bag)
                g3.pick_ship(slot)
                cell = self.best_ship_cell(g3, g3.player(), list(L.LONGSHIPS[g3.ocean_ships[slot]][0]))
                g3.place_ship(cell)
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
                vals[k] = E.outcome(g, me)[0]          # known exactly
            else:
                todo.append(k)
        if todo:
            enc = [E.encode(games[k], me) for k in todo]
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
        # Stage 1: every placement, judged quickly; keep the best few
        if g.phase == "place":
            firsts = [(opt,) + self._place(g, opt) for opt in self.placement_options(g)]
            firsts = self._shortlist(firsts, me, self.TOP_FULL)
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
                seconds = [(o2,) + self._place(g2x, o2) for o2 in self.placement_options(g2x)]
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
    Both can be combined: 'nn:v1@2+2'."""
    model = name.split(":", 1)[-1]
    depth = 1
    if "+" in model:
        model, d = model.split("+")
        depth = int(d)
    ship_cost = 0.0
    if "@" in model:
        model, cost = model.split("@")
        ship_cost = float(cost)
    return NNBot(rng, model, explore, ship_cost, depth)
