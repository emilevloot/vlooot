"""
The neural network that judges a Looot position (PyTorch).

It predicts, for the player "me" of the encoding (nn_encode.py):
  margin  my final score minus the opponent's (in units of 20 points)
  win     how likely I am to win (as a logit; sigmoid gives the chance)

Building block: a hex convolution. For every space it combines the space
itself with its 6 neighbours, each direction with its own weights - like
the 3x3 window of an image network, but for hexagons.

A hex convolution only sees the neighbours, so after 6 of them a space knows
about spaces up to 6 steps away. To see the WHOLE board, every block also
takes the average and the maximum of each channel over all spaces ("global
pooling") and adds a mix of those to every space.

Two kinds of network (ARCHS):
  value   runs 1-4: land, fjords and global numbers combined at the end
  three   run 5: three parts (fjord, board, longships) passing each other
          messages of a fixed shape - see ThreePartNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_encode as E

MARGIN_SCALE = 20.0     # the network predicts margin / 20 (easier numbers)


class HexConv(nn.Module):
    """out[space] = W0 * x[space] + sum over directions d of Wd * x[neighbour d]"""

    def __init__(self, cin, cout, nb):
        super().__init__()
        self.register_buffer("nb", torch.as_tensor(nb, dtype=torch.long))
        self.w = nn.Parameter(torch.randn(7, cin, cout) * (2.0 / (7 * cin)) ** 0.5)
        self.b = nn.Parameter(torch.zeros(cout))

    def forward(self, x):                      # x: [batch, spaces, cin]
        pad = torch.zeros_like(x[:, :1])       # "no neighbour" = zeros
        xp = torch.cat([x, pad], dim=1)
        nbx = xp[:, self.nb]                   # [batch, spaces, 6, cin]
        out = x @ self.w[0] + torch.einsum("bndc,dco->bno", nbx, self.w[1:])
        return out + self.b


class Block(nn.Module):
    """One residual block: a hex convolution (the neighbours) plus global
    pooling (the whole board), both ADDED to what the space already had."""

    def __init__(self, width, nb):
        super().__init__()
        self.conv = HexConv(width, width, nb)
        self.pool = nn.Linear(2 * width, width)

    def forward(self, h, mask=None):           # h: [batch, spaces, width]
        if mask is None:
            whole = torch.cat([h.mean(1), h.amax(1)], dim=1)   # [batch, 2*width]
            return F.relu(h + self.conv(h) + self.pool(whole).unsqueeze(1))
        # mask [batch, spaces, 1]: 0 for spaces not in this game - they
        # stay 0 and don't count in the average
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        whole = torch.cat([mean, h.amax(1)], dim=1)
        return F.relu(h + self.conv(h) + self.pool(whole).unsqueeze(1)) * mask


class Board(nn.Module):
    """Hex convolutions over one board, squeezed to `out` numbers."""

    BLOCKS = 4

    def __init__(self, cin, width, squeeze, nb, n_spaces, out):
        super().__init__()
        self.c1 = HexConv(cin, width, nb)
        self.blocks = nn.ModuleList(Block(width, nb) for _ in range(self.BLOCKS))
        self.cout = HexConv(width, squeeze, nb)
        self.fc = nn.Linear(n_spaces * squeeze, out)

    def parts(self, x, mask=None):
        """(the board summed up in `out` numbers, and every space's own
        numbers [batch, spaces, squeeze]). mask [batch, spaces, 1]: which
        spaces are in the game (None: all)."""
        if mask is not None:
            x = x * mask
        h = F.relu(self.c1(x))
        if mask is not None:
            h = h * mask
        for b in self.blocks:
            h = b(h, mask)
        cells = F.relu(self.cout(h))
        if mask is not None:
            cells = cells * mask
        return F.relu(self.fc(cells.flatten(1))), cells

    def forward(self, x):
        return self.parts(x)[0]


def _scaling_buffers(m):
    """Input scaling (mean / spread per feature), filled in from the data."""
    for name, n in (("land", m.E.LAND_F), ("fjord", m.E.FJORD_F), ("glob", m.E.GLOB_F)):
        m.register_buffer(name + "_mu", torch.zeros(n))
        m.register_buffer(name + "_sd", torch.ones(n))


class ValueNet(nn.Module):
    """The network of runs 1-4: land, fjords and the global numbers each get
    their own layers, and are combined at the very end."""

    ARCH = "value"
    E = E                      # the encoding (nn_encode; mp_encode for 2-4 players)

    def __init__(self, dropout=0.0):
        super().__init__()
        # dropout: while training, randomly switch off this share of the
        # neurons so the network can't memorise games (off while playing)
        self.drop = nn.Dropout(dropout)
        self.land = Board(self.E.LAND_F, 48, 16, self.E.LAND_NB, self.E.N_LAND, 128)
        self.fjord = Board(self.E.FJORD_F, 32, 12, self.E.FJORD_NB, self.E.N_FJORD, 64)
        self.glob = nn.Linear(self.E.GLOB_F, 64)
        self.fc1 = nn.Linear(128 + 64 + 64 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.margin = nn.Linear(128, 1)
        self.win = nn.Linear(128, 1)
        _scaling_buffers(self)

    def forward(self, land, fjord, glob):
        """land [B,55,38], fjord [B,2,37,35], glob [B,158] (raw numbers).
        Returns (margin / 20, win logit), each [B]."""
        return self.explain(land, fjord, glob)[:2]

    def explain(self, land, fjord, glob):
        """(margin / 20, win logit, the messages between the parts)"""
        a, cells = self.land.parts((land.float() - self.land_mu) / self.land_sd, self.land_mask(land))
        fjord = (fjord.float() - self.fjord_mu) / self.fjord_sd
        # every fjord through the same layers: mine first, then the others
        bs, fcs = zip(*[self.fjord.parts(fjord[:, k]) for k in range(fjord.shape[1])])
        b, c, fb, fc = self.seats(torch.stack(bs, 1), torch.stack(fcs, 1), glob)
        return self.head(land, glob, a, cells, b, c, (fb, fc))

    def land_mask(self, land):
        """[B, spaces, 1]: 1 for spaces in the game (multi-player encoding
        only; with 2 players every space is in the game: None)."""
        if not hasattr(self.E, "LF_INPLAY"):
            return None
        return land[:, :, self.E.LF_INPLAY:self.E.LF_INPLAY + 1].float()

    def seats(self, bs, fcs, glob):
        """My fjord, and "the opponent's": with 2 players simply the other
        one; with more, the largest number over the opponents in the game
        (per number: the most advanced opponent fjord in that respect).
        bs [B, seats, 64] fjord summaries, fcs [B, seats, 37, 12] their spaces."""
        if bs.shape[1] == 2:
            return bs[:, 0], bs[:, 1], fcs[:, 0], fcs[:, 1]
        on = glob[:, self.E.PRESENT_COLS] > 0                   # [B, 3] seat in the game?
        c = _masked_max(bs[:, 1:], on.unsqueeze(2).expand(-1, -1, bs.shape[2]))
        fc = _masked_max(fcs[:, 1:], on[:, :, None, None].expand(-1, -1, *fcs.shape[2:]))
        return bs[:, 0], c, fcs[:, 0], fc

    def head(self, land, glob, a, cells, b, c, fj_cells=None):
        """Everything after the board layers. land / glob: the raw numbers;
        a, cells: the land part's output; b, c: my / the opponent's fjord
        (fj_cells: their spaces' own numbers).
        (Split off so the GPU player can run each board only once.)"""
        g = (glob.float() - self.glob_mu) / self.glob_sd
        d = F.relu(self.glob(g))
        h = self.drop(torch.cat([a, b, c, d], dim=1))
        h = self.drop(F.relu(self.fc1(h)))
        h = F.relu(self.fc2(h))
        return self.margin(h).squeeze(1), self.win(h).squeeze(1), {}


# ---------------------------------------------------------------------------
# The network in three parts (run 5)
# ---------------------------------------------------------------------------

ITEMS = E.SITE_ITEMS                   # wood sheep gold axe house watchtower castle
N_ITEMS = len(ITEMS)
SHIP_F = 4 + N_ITEMS + 4 + N_ITEMS + 4 + 2


def _masked_max(x, mask):
    """Largest x where mask is on; 0 where it is on nowhere."""
    m = torch.where(mask, x, torch.full_like(x, -1e9)).amax(1)
    return torch.where(mask.any(1), m, torch.zeros_like(m))


def _masked_mean(x, mask):
    mf = mask.float()
    return (x * mf).sum(1) / mf.sum(1).clamp(min=1)


class ThreePartNet(nn.Module):
    """The network in three parts that pass each other messages of a fixed
    shape, trained together on the final result:

      1. FJORD part   looks at both fjords and the clock, and gives a VALUE
                      for every item (wood ... castle): what one more is worth
                      to me now, and (same layers) to the opponent
      2. BOARD part   looks at the land. For every free space: what I would
                      get there x what that is worth = "a Viking here is
                      worth ...". Also: how much of each item is on offer.
      3. SHIP part    for every longship in the ocean: how well it fits,
                      from what it needs, its bonus, the item values and
                      what the board offers (the same layers for every ship)

    The messages (item values, placement values, ship fits) can be read out
    with explain(): the game shows them, so you can see what it wants."""

    ARCH = "three"
    E = E

    def __init__(self, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.land = Board(self.E.LAND_F, 48, 16, self.E.LAND_NB, self.E.N_LAND, 128)
        self.fjord = Board(self.E.FJORD_F, 32, 12, self.E.FJORD_NB, self.E.N_FJORD, 64)
        self.glob = nn.Linear(self.E.GLOB_F, 64)
        # 1. fjord part: [this fjord, the other fjord, clock] -> a value per item
        self.values = nn.Sequential(nn.Linear(64 + 64 + 2, 64), nn.ReLU(), nn.Linear(64, N_ITEMS))
        # 2. board part: per space [its own numbers, worth for me, for the
        #    opponent, free?] -> a score for me and one for the opponent
        self.cell = nn.Sequential(nn.Linear(16 + 3, 16), nn.ReLU(), nn.Linear(16, 2))
        # 3. ship part: per longship -> how well it fits
        self.ship = nn.Sequential(nn.Linear(SHIP_F, 32), nn.ReLU(), nn.Linear(32, 1))
        n_msg = 2 * N_ITEMS + 8 + 2 * N_ITEMS + 4
        self.fc1 = nn.Linear(128 + 64 + 64 + 64 + n_msg, 256)
        self.fc2 = nn.Linear(256, 128)
        self.margin = nn.Linear(128, 1)
        self.win = nn.Linear(128, 1)
        self.register_buffer("bonus_on", torch.tensor(self.E.BONUS_ON))
        # --- the PROPOSALS ("policy"): which moves look promising ---
        # They are needed once per turn (the judgement: for every candidate),
        # so they get hex layers of their own: pol_land over the land,
        # pol_fjord over my fjord. They also read the three parts' numbers.
        self.pol_land = Board(self.E.LAND_F, 32, 16, self.E.LAND_NB, self.E.N_LAND, 16)
        self.pol_fjord = Board(self.E.FJORD_F, 32, 12, self.E.FJORD_NB, self.E.N_FJORD, 16)
        # plaats-kaart: per land space [own 16, the board part's 16, worth
        # for me / the opponent, free?, stackable?] -> plain / double / stacked
        self.pol_place = nn.Sequential(nn.Linear(16 + 16 + 4, 32), nn.ReLU(), nn.Linear(32, 3))
        # waar-kaart: per space of MY fjord [own 12, the fjord part's 12, my
        # item values] -> each of the 7 items, and a new longship
        self.pol_tile = nn.Sequential(nn.Linear(12 + 12 + N_ITEMS, 32), nn.ReLU(),
                                      nn.Linear(32, N_ITEMS + 1))
        # schip-keuze: the ship part's fit of every ocean longship, and "none"
        self.pol_none = nn.Linear(128, 1)
        _scaling_buffers(self)

    forward = ValueNet.forward
    seats = ValueNet.seats
    land_mask = ValueNet.land_mask

    def explain(self, land, fjord, glob):
        m, w, told = ValueNet.explain(self, land, fjord, glob)
        inner = told.pop("_inner")
        if self.training or getattr(self, "with_proposals", False):
            told.update(self.proposals(land, fjord, inner))
        return m, w, told

    def proposals(self, land, fjord, inner):
        """The three maps (raw scores; softmax over the legal moves):
        pol_place [B, 55, 3], pol_tile [B, 37, 8], pol_ship [B, 6]."""
        cells, place_me, place_opp, free, stack, fb, val_me, fits, h = inner
        own_l = self.pol_land.parts((land.float() - self.land_mu) / self.land_sd,
                                    self.land_mask(land))[1]
        own_f = self.pol_fjord.parts((fjord[:, 0].float() - self.fjord_mu) / self.fjord_sd)[1]
        place = self.pol_place(torch.cat([own_l, cells, place_me.unsqueeze(2),
                                          place_opp.unsqueeze(2), free.unsqueeze(2),
                                          stack.unsqueeze(2)], 2))
        tile = self.pol_tile(torch.cat([own_f, fb, val_me.unsqueeze(1).expand(-1, fb.shape[1], -1)], 2))
        ship = torch.cat([fits, self.pol_none(h)], 1)
        return {"pol_place": place, "pol_tile": tile, "pol_ship": ship}

    def head(self, land, glob, a, cells, b, c, fj_cells=None):
        land = land.float()
        glob = glob.float()
        g = (glob - self.glob_mu) / self.glob_sd
        clock = g[:, self.E.CLOCK_COLS]                                    # [B, 2]
        # --- 1. fjord part: item values for me and for the opponent ---
        val_me = self.values(torch.cat([b, c, clock], 1))      # [B, 7]
        val_opp = self.values(torch.cat([c, b, clock.flip(1)], 1))
        # --- 2. board part ---
        free = land[:, :, self.E.LF_FREE]                           # [B, 55]
        res = land[:, :, self.E.RES_COLS] * free.unsqueeze(2)         # the resource you'd get
        got_me = torch.cat([res, land[:, :, self.E.GAIN_BLD[0]]], 2)  # [B, 55, 7] items I'd get
        got_opp = torch.cat([res, land[:, :, self.E.GAIN_BLD[1]]], 2)
        place_me = (got_me * val_me.unsqueeze(1)).sum(2)       # [B, 55] a Viking here is worth
        place_opp = (got_opp * val_opp.unsqueeze(1)).sum(2)
        score = self.cell(torch.cat([cells, place_me.unsqueeze(2), place_opp.unsqueeze(2),
                                     free.unsqueeze(2)], 2))    # [B, 55, 2]
        on = free > 0
        pooled = torch.stack([_masked_max(place_me, on), _masked_mean(place_me, on),
                              _masked_max(place_opp, on), _masked_mean(place_opp, on),
                              _masked_max(score[:, :, 0], on), _masked_mean(score[:, :, 0], on),
                              _masked_max(score[:, :, 1], on), _masked_mean(score[:, :, 1], on)], 1)
        offer_me = got_me.sum(1) / 10                          # [B, 7] items on offer
        offer_opp = got_opp.sum(1) / 10
        # --- 3. ship part: the same small layers for every longship ---
        need = glob[:, self.E.SHIP_NEED]                              # [B, 5, 4]
        bonus = glob[:, self.E.SHIP_BONUS] * self.bonus_on            # [B, 5, 7]
        present = need.sum(2) > 0                              # [B, 5]
        fits = []
        for val, offer, clk in ((val_me, offer_me, clock), (val_opp, offer_opp, clock.flip(1))):
            v, o = val.unsqueeze(1), offer.unsqueeze(1)
            x = torch.cat([need, bonus, need * v[:, :, :4], bonus * v, need * o[:, :, :4],
                           clk.unsqueeze(1).expand(-1, 5, -1)], 2)          # [B, 5, SHIP_F]
            fits.append(self.ship(x).squeeze(2))                            # [B, 5]
        ships = torch.stack([_masked_max(fits[0], present), _masked_mean(fits[0], present),
                             _masked_max(fits[1], present), _masked_mean(fits[1], present)], 1)
        # --- everything together ---
        d = F.relu(self.glob(g))
        msg = torch.cat([val_me, val_opp, pooled, offer_me, offer_opp, ships], 1)
        h = self.drop(torch.cat([a, b, c, d, msg], 1))
        h = self.drop(F.relu(self.fc1(h)))
        h = F.relu(self.fc2(h))
        told = {"values_me": val_me, "values_opp": val_opp, "place_me": place_me,
                "ships_me": torch.where(present, fits[0], torch.zeros_like(fits[0])),
                "present": present}
        # what the proposals read (see proposals()); .detach(): learning the
        # proposals doesn't change the three parts - the judgement stays as it was
        told["_inner"] = (cells.detach(), place_me.detach(), place_opp.detach(), free,
                          land[:, :, self.E.LF_STACK], fj_cells[0].detach(), val_me.detach(),
                          fits[0].detach(), h.detach())
        return self.margin(h).squeeze(1), self.win(h).squeeze(1), told


# ---------------------------------------------------------------------------
# The network with attention (a transformer on top of the hex layers)
# ---------------------------------------------------------------------------

# tokens: the land spaces, 2 fjords (mine, the opponents'), 5 longships, "global"


class AttnLayer(nn.Module):
    """One transformer layer. Every token (a space, a longship, the global
    token) asks every other token for information: it makes a question
    (query), every token offers a label (key) and a content (value); the
    better a label matches the question, the more of that content it takes
    ("attention"). 2 heads = 2 different questions at once. Then a small
    layer per token. Both steps are ADDED to the token (residual)."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, x):                      # x: [batch, tokens, d]
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)            # [B, heads, T, D/heads]
        x = x + self.out(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(F.relu(self.ff1(self.ln2(x))))


class AttnNet(nn.Module):
    """The hex layers of the value network (local patterns on each board),
    then 2 transformer layers over ALL spaces of the land and both fjords,
    the 5 longships in the ocean and one global token: any space can take
    information from any other, also on another board (a landscape space
    can "see" which of my longships still misses its resource)."""

    ARCH = "attn"
    E = E
    D, HEADS, FF, LAYERS = 32, 2, 64, 2         # (bigger: much slower, and too big for the GPU)

    def __init__(self, dropout=0.0):
        super().__init__()
        d = self.D
        self.drop = nn.Dropout(dropout)
        self.land = Board(self.E.LAND_F, 48, 16, self.E.LAND_NB, self.E.N_LAND, 128)
        self.fjord = Board(self.E.FJORD_F, 32, 12, self.E.FJORD_NB, self.E.N_FJORD, 64)
        self.glob = nn.Linear(self.E.GLOB_F, 64)
        # turning everything into tokens of d numbers, plus where/what it is
        self.tok_land = nn.Linear(16, d)
        self.tok_fjord = nn.Linear(12, d)
        self.tok_ship = nn.Linear(4 + len(self.E.SITE_ITEMS), d)
        self.tok_glob = nn.Linear(self.E.GLOB_F, d)
        self.pos = nn.Parameter(torch.randn(self.E.N_LAND + 2 * self.E.N_FJORD + 6, d) * 0.02)   # one per token place
        self.layers = nn.ModuleList(AttnLayer(d, self.HEADS, self.FF) for _ in range(self.LAYERS))
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(128 + 64 + 64 + 64 + 2 * d, 256)
        self.fc2 = nn.Linear(256, 128)
        self.margin = nn.Linear(128, 1)
        self.win = nn.Linear(128, 1)
        self.register_buffer("bonus_on", torch.tensor(self.E.BONUS_ON))
        _scaling_buffers(self)

    forward = ValueNet.forward
    explain = ValueNet.explain
    seats = ValueNet.seats
    land_mask = ValueNet.land_mask

    def head(self, land, glob, a, cells, b, c, fj_cells=None):
        glob = glob.float()
        g = (glob - self.glob_mu) / self.glob_sd
        need = glob[:, self.E.SHIP_NEED]                                  # [B, 5, 4]
        bonus = glob[:, self.E.SHIP_BONUS] * self.bonus_on                # [B, 5, 7]
        fme, fopp = fj_cells
        x = torch.cat([self.tok_land(cells), self.tok_fjord(fme), self.tok_fjord(fopp),
                       self.tok_ship(torch.cat([need, bonus], 2) / 3),
                       self.tok_glob(g).unsqueeze(1)], 1) + self.pos
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        d = F.relu(self.glob(g))
        h = self.drop(torch.cat([a, b, c, d, x[:, -1], x.mean(1)], 1))
        h = self.drop(F.relu(self.fc1(h)))
        h = F.relu(self.fc2(h))
        return self.margin(h).squeeze(1), self.win(h).squeeze(1), {}


# ---------------------------------------------------------------------------
# The same networks for 2-4 players (encoding mp_encode.py): arch name + "4"
# ---------------------------------------------------------------------------

import mp_encode  # noqa: E402


class ValueNet4(ValueNet):
    ARCH = "value4"
    E = mp_encode


class ThreePartNet4(ThreePartNet):
    ARCH = "three4"
    E = mp_encode


class AttnNet4(AttnNet):
    ARCH = "attn4"
    E = mp_encode


ARCHS = {m.ARCH: m for m in (ValueNet, ThreePartNet, AttnNet, ValueNet4, ThreePartNet4, AttnNet4)}


def load(path, dropout=0.0):
    """A saved network (models/<name>.pt), whichever kind it is."""
    d = torch.load(path, map_location="cpu")
    model = ARCHS[d.get("arch", "value")](dropout)
    own = model.state_dict()
    # proposal layers of another shape (an older version of them) start fresh
    state = {k: v for k, v in d["state"].items()
             if not (k.startswith("pol_") and k in own and own[k].shape != v.shape)}
    missing, extra = model.load_state_dict(state, strict=False)
    # networks from before the proposals: those layers start fresh
    assert not extra and all(k.startswith("pol_") for k in missing), (missing, extra)
    return model


def count_weights(model):
    return sum(p.numel() for p in model.parameters())


def transfer_to_mp(src, stats=None):
    """A new 2-4-player network of the same kind as the 2-player network
    `src`, starting from everything src learned:
      - the hex layers of land and fjord: copied (the extra land feature
        "in play" starts with weight 0)
      - the land's last layer: the weights of boards 1-2 and the ocean go to
        the same spaces; boards 3-4 start at 0 (to be learned)
      - the global numbers: every 2-player number keeps its weights; the new
        multi-player numbers start at 0
      - all other layers: copied as they are
    stats: {"land_mu", "land_sd", "glob_mu", "glob_sd"} measured on
    multi-player data, used for the NEW features only."""
    dst = ARCHS[src.ARCH + "4"]()
    E2, EM = src.E, dst.E
    ss, ds = src.state_dict(), dst.state_dict()
    cell_map = [EM.LAND_INDEX[c] for c in E2.LAND_CELLS]
    gmap = torch.as_tensor(EM.GLOB_FROM_2P)
    new_glob = torch.ones(EM.GLOB_F, dtype=torch.bool)
    new_glob[gmap] = False
    out = {}
    for k, v in ds.items():
        if k not in ss or k.endswith(".nb"):          # the neighbour tables are the new ones
            continue
        o = ss[k]
        if o.shape == v.shape:
            out[k] = o.clone()
        elif k.endswith("land.c1.w"):                 # [7, 38, 48] -> [7, 39, 48] (also pol_land)
            w = torch.zeros_like(v)
            w[:, :o.shape[1]] = o
            out[k] = w
        elif k.endswith("land.fc.weight"):            # [128, 55*16] -> [128, 105*16] (also pol_land)
            w = torch.zeros_like(v)
            sq = o.shape[1] // E2.N_LAND
            for i, j in enumerate(cell_map):
                w[:, j * sq:(j + 1) * sq] = o[:, i * sq:(i + 1) * sq]
            out[k] = w
        elif k in ("glob.weight", "tok_glob.weight"):
            w = torch.zeros_like(v)
            w[:, gmap] = o
            out[k] = w
        elif k in ("land_mu", "land_sd"):
            w = v.clone()
            w[:o.shape[0]] = o
            if stats is not None:
                w[o.shape[0]:] = torch.as_tensor(stats[k][o.shape[0]:])
            out[k] = w
        elif k in ("glob_mu", "glob_sd"):
            w = v.clone()
            w[gmap] = o
            if stats is not None:
                w[new_glob] = torch.as_tensor(stats[k])[new_glob]
            out[k] = w
        elif k == "pos":                              # attention: one row per token place
            w = v.clone()
            nl2, nlm, nf = E2.N_LAND, EM.N_LAND, E2.N_FJORD
            for i, j in enumerate(cell_map):
                w[j] = o[i]
            w[nlm:] = o[nl2:]                         # fjords, longships, global: same order
            out[k] = w
    ds.update(out)
    dst.load_state_dict(ds)
    return dst
