"""
Looot for 2 players as numpy arrays, compiled with Numba: a fast copy of the
game rules and of the neural-network player, for training. Not for playing
by people - looot.py stays the real game, and test_fastgame.py checks that
both give exactly the same results.

The whole changing state of a game is ONE int32 array `s` (see the S_*
offsets below), so copying a game is a single small memory copy. What never
changes during a game (terrain, which spaces are towers, what the
construction sites need, which rules are on) is a second array, `gs`.

Every rule is a function compiled by Numba (@njit): plain loops over
numbers, turned into machine code, so they run about as fast as C. Numba
can't handle dicts of tuples or Python objects, which is why the game is
written again here as arrays.

Self-play:  selfplay_game(seed, layout, rules, net, ...)  plays one game
between two copies of the network player and returns the encoded positions,
in the same format as gen_data.py.
"""

import glob
import hashlib
import os

import numpy as np
from numba import njit

import looot as L
import nn_encode as E

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Fixed geometry (2 players: boards 1 and 2 plus the ocean)
# ---------------------------------------------------------------------------

NL = E.N_LAND                        # 55 spaces: 50 on the boards + 5 ocean
NF = E.N_FJORD                       # 37 fjord spaces
NB = NL - len(L.OCEAN_CELLS)         # 50 board spaces
assert [tuple(c) for c in E.LAND_CELLS[NB:]] == list(L.OCEAN_CELLS)
LAND_NB = np.ascontiguousarray(E.LAND_NB, dtype=np.int64)     # padding = NL
FJ_NB = np.ascontiguousarray(E.FJORD_NB, dtype=np.int64)      # padding = NF

MAXT = 16            # watchtowers on the gameboard
MAXP = 48            # tiles waiting to be placed after one Viking
MAXO = 160           # placement options in one turn

# terrain codes = index in E.TERRAINS; the ocean gets 7
assert E.TERRAINS == ["forest", "field", "mountain", "battlefield",
                      "house", "watchtower", "castle"]
T_HOUSE, T_TOWER, T_CASTLE, T_OCEAN = 4, 5, 6, 7
# fjord codes: 0 empty, 1..7 = item type + 1, 8 longship, 9 construction site
assert E.SITE_ITEMS == ["wood", "sheep", "gold", "axe", "house", "watchtower", "castle"]
FJ_SHIP, FJ_SITE = 8, 9

CAT_ITEM = np.array([E.SITE_ITEMS.index(c) for c in E.CATS], dtype=np.int64)
BASE = np.array([L.BASE_VALUE[c] for c in E.CATS], dtype=np.int64)
LS_NEED = np.zeros((len(L.LONGSHIPS), 4), dtype=np.int64)
LS_CAT = np.zeros(len(L.LONGSHIPS), dtype=np.int64)
LS_BONUS = np.zeros(len(L.LONGSHIPS), dtype=np.int64)
for _i, (_need, _cat, _bonus) in enumerate(L.LONGSHIPS):
    for _r in _need:
        LS_NEED[_i, E.RES.index(_r)] += 1
    LS_CAT[_i] = E.CATS.index(_cat)
    LS_BONUS[_i] = _bonus
T_AXES = np.array([a for a, v in L.TROPHIES], dtype=np.int64)
T_VP = np.array([v for a, v in L.TROPHIES], dtype=np.int64)
SITE_KINDS = list(L.FJORD_SITE_SPOTS)                     # altar, port, jarl
SITE_CELL = np.array([E.FJORD_INDEX[L.FJORD_SITE_SPOTS[k]] for k in SITE_KINDS], dtype=np.int64)
SITE_VP = np.array([L.SITE_VP[k] for k in SITE_KINDS], dtype=np.int64)
SITE_OF_CELL = np.full(NF, -1, dtype=np.int64)
SITE_OF_CELL[SITE_CELL] = np.arange(len(SITE_KINDS))
# board spaces sorted by (q, r): the order looot.py uses for captures
COORD_ORDER = np.array(sorted(range(NB), key=lambda i: tuple(E.LAND_CELLS[i])), dtype=np.int64)
COORD_RANK = np.zeros(NL, dtype=np.int64)
COORD_RANK[COORD_ORDER] = np.arange(NB)
# rank of each space's "q,r" text key (nn_encode counts tower links by it)
_keys = [L.key(tuple(E.LAND_CELLS[i])) for i in range(NB)]
KEY_RANK = np.zeros(NL, dtype=np.int64)
for _rank, _i in enumerate(sorted(range(NB), key=lambda i: _keys[i])):
    KEY_RANK[_i] = _rank

# ---------------------------------------------------------------------------
# State layout: offsets into the int32 state array `s`
# ---------------------------------------------------------------------------

S_VIK = 0                       # [2][NL] Vikings per player per space
S_STOCK = S_VIK + 2 * NL        # [NL] tiles left on a building
S_OCEAN = S_STOCK + NL          # [5] longship id, or -1
S_BAG = S_OCEAN + 5             # [30] the bag; the top is S_BAG + bag size - 1
S_BAGN = S_BAG + 30
S_VLEFT = S_BAGN + 1            # [2] Vikings left
S_SHIELD = S_VLEFT + 2          # [2][3] extra, occupy, double
S_TROPHY = S_SHIELD + 6         # [2] trophy index, or -1
S_TOWNER = S_TROPHY + 2         # [5] owner of each trophy, or -1
S_CASTLE = S_TOWNER + 5         # [2][NL] castle tiles taken per castle space
S_PAIRS = S_CASTLE + 2 * NL     # [2][MAXT] bitmask of linked towers per tower
S_FJ = S_PAIRS + 2 * MAXT       # [2][NF] fjord codes
S_FJSHIP = S_FJ + 2 * NF        # [2][NF] longship id on a fjord space
S_FJFLAG = S_FJSHIP + 2 * NF    # [2][NF] longship filled / site done
S_CUR = S_FJFLAG + 2 * NF       # player to move
S_OVER = S_CUR + 1              # game over
S_PHASE = S_OVER + 1
S_PLACED = S_PHASE + 1          # Vikings placed this turn
S_TOOK = S_PLACED + 1           # took a longship this turn
S_LEN = S_TOOK + 1
PH_PLACE, PH_TILES, PH_ACTIONS, PH_OVER = 0, 1, 2, 3

# game-static array `gs` (int64)
G_TERR = 0                      # [NL] terrain code
G_TSLOT = G_TERR + NL           # [NL] tower slot, or -1
G_TCELL = G_TSLOT + NL          # [MAXT] space of each tower slot, or -1
G_NT = G_TCELL + MAXT           # number of towers
G_SITENEED = G_NT + 1           # [2][3][7] what each construction site needs
G_RULES = G_SITENEED + 2 * 3 * 7   # [5] buildings, sites, longships, shields, trophies
G_LEN = G_RULES + 5
R_BUILD, R_SITES, R_LONG, R_SHIELDS, R_TROPHIES = 0, 1, 2, 3, 4
RULE_KEYS = ["buildings", "sites", "longships", "shields", "trophies"]

# bot weights (passed in as an array, so changed weights need no recompile)
BW_KEYS = ["tile_need", "tile_block", "tile_open", "shipcell_match", "shipcell_room"]
KEEP = np.array([1.0, 1.0, 2.0, 1.5, 2.0, 3.0, 5.0])   # Bot.KEEP_ORDER per item type


def bot_weights(w=None):
    w = dict(L.BOT_WEIGHTS, **(w or {}))
    return np.array([w[k] for k in BW_KEYS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Encoding layout (the same numbers as nn_encode.encode)
# ---------------------------------------------------------------------------

LAND_F, FJORD_F, GLOB_F = E.LAND_F, E.FJORD_F, E.GLOB_F
LF_TERRAIN, LF_OCEAN, LF_SHIP = E.LF_TERRAIN, E.LF_OCEAN, E.LF_SHIP
LF_MY_VIK, LF_OPP_VIK, LF_STOCK = E.LF_MY_VIK, E.LF_OPP_VIK, E.LF_STOCK
LF_MY_CHAIN, LF_OPP_CHAIN = E.LF_MY_CHAIN, E.LF_OPP_CHAIN
LF_MY_CASTLE, LF_OPP_CASTLE = E.LF_MY_CASTLE, E.LF_OPP_CASTLE
LF_MY_TOWERS, LF_OPP_TOWERS, LF_ANCHOR = E.LF_MY_TOWERS, E.LF_OPP_TOWERS, E.LF_ANCHOR
FF_EMPTY, FF_RES, FF_BLD = E.FF_EMPTY, E.FF_RES, E.FF_BLD
FF_SHIP, FF_SHIP_DONE, FF_SHIP_MISSING, FF_SHIP_BONUS = (
    E.FF_SHIP, E.FF_SHIP_DONE, E.FF_SHIP_MISSING, E.FF_SHIP_BONUS)
FF_SITE, FF_SITE_DONE, FF_SITE_MISSING, FF_SITE_VP = (
    E.FF_SITE, E.FF_SITE_DONE, E.FF_SITE_MISSING, E.FF_SITE_VP)
FF_ROOM, FF_FILLABLE, FF_SLACK = E.FF_ROOM, E.FF_FILLABLE, E.FF_SLACK

G_ = E.G
GI_VIK = np.array([G_["vik_me"], G_["vik_opp"]])
GI_SHIELD = np.array([[G_["shield_%s_%s" % (w, s)] for s in ("extra", "occupy", "double")]
                      for w in ("me", "opp")])
GI_AXES = np.array([G_["axes_me"], G_["axes_opp"]])
GI_COUNT = np.array([[G_["count_%s_%s" % (c, w)] for c in E.CATS] for w in ("me", "opp")])
GI_VALUE = np.array([[G_["value_%s_%s" % (c, w)] for c in E.CATS] for w in ("me", "opp")])
GI_SITES = np.array([G_["sites_me"], G_["sites_opp"]])
GI_UNFILLED = np.array([G_["unfilled_me"], G_["unfilled_opp"]])
GI_TOTAL = np.array([G_["total_me"], G_["total_opp"]])
GI_ROOM = np.array([G_["room_me"], G_["room_opp"]])
GI_SPOTS = np.array([G_["ship_spots_me"], G_["ship_spots_opp"]])
GI_TROPHY = np.array([[G_["trophy%d_me" % i], G_["trophy%d_opp" % i]] for i in range(5)])
GI_OCEAN_RES = np.array([[G_["ocean%d_%s" % (k, r)] for r in E.RES] for k in range(5)])
GI_OCEAN_BONUS = np.array([[G_["ocean%d_bonus_%s" % (k, c)] for c in E.CATS] for k in range(5)])
GI_ANCHORS = np.array([G_["anchors_" + r] for r in E.RES])
GI_TOMOVE, GI_BAG = G_["to_move_me"], G_["bag"]


def _check_cache():
    """Numba keeps compiled code on disk. It notices when THIS file changes,
    but not when nn_encode.py or looot.py change numbers used here, so we
    fingerprint those numbers and clear the compiled code when they differ."""
    sig = hashlib.sha1(repr((E.GLOB_NAMES, E.LAND_F, E.FJORD_F, E.ENCODING_VERSION,
                             L.LONGSHIPS, L.TROPHIES, L.BASE_VALUE, L.SITE_VP,
                             L.FJORD_SITE_SPOTS, L.OCEAN_CELLS, L.BOARD_PLACEMENT,
                             L.BOARD_SHAPE, sorted(L.FJORD_COLUMNS.items()))).encode()).hexdigest()
    cache = os.path.join(HERE, "__pycache__")
    stamp = os.path.join(cache, "fastgame.sig")
    old = open(stamp).read() if os.path.exists(stamp) else None
    if old != sig:
        for f in glob.glob(os.path.join(cache, "fastgame.*.nb[ic]")):
            try:
                os.remove(f)
            except OSError:
                pass
        os.makedirs(cache, exist_ok=True)
        with open(stamp, "w") as f:
            f.write(sig)


_check_cache()


# ---------------------------------------------------------------------------
# From a looot.Game to arrays (every game is set up by looot.py)
# ---------------------------------------------------------------------------

PHASES = {"place": PH_PLACE, "tiles": PH_TILES, "actions": PH_ACTIONS, "over": PH_OVER}


def from_game(g):
    """(s, gs) arrays for a 2-player looot.Game. Tiles still waiting to be
    placed (g.pending) are not part of `s`."""
    assert len(g.players) == 2, "fastgame is for 2-player games"
    s = np.zeros(S_LEN, dtype=np.int32)
    gs = np.zeros(G_LEN, dtype=np.int64)
    gs[G_TERR:G_TERR + NL] = T_OCEAN
    gs[G_TSLOT:G_TSLOT + NL] = -1
    gs[G_TCELL:G_TCELL + MAXT] = -1
    nt = 0
    for i in range(NB):
        code = E.TERRAINS.index(g.land[E.LAND_CELLS[i]])
        gs[G_TERR + i] = code
        if code == T_TOWER:
            assert nt < MAXT, "too many watchtowers"
            gs[G_TSLOT + i] = nt
            gs[G_TCELL + nt] = i
            nt += 1
    gs[G_NT] = nt
    for k, r in enumerate(RULE_KEYS):
        gs[G_RULES + k] = 1 if g.rules[r] else 0
    for p, pl in enumerate(g.players):
        for c, it in pl.fjord.items():
            if it["kind"] == "site":
                k = SITE_KINDS.index(it["type"])
                for t in it["need"]:
                    gs[G_SITENEED + (p * 3 + k) * 7 + E.SITE_ITEMS.index(t)] += 1

    for c, v in g.vikings.items():
        i = E.LAND_INDEX[c]
        s[S_VIK + i] = v.count(0)
        s[S_VIK + NL + i] = v.count(1)
    for c, n in g.stock.items():
        s[S_STOCK + E.LAND_INDEX[c]] = n
    for k, sid in enumerate(g.ocean_ships):
        s[S_OCEAN + k] = -1 if sid is None else sid
    s[S_BAG:S_BAG + len(g.bag)] = g.bag
    s[S_BAGN] = len(g.bag)
    for p, pl in enumerate(g.players):
        s[S_VLEFT + p] = pl.vikings_left
        for k, name in enumerate(("extra", "occupy", "double")):
            s[S_SHIELD + p * 3 + k] = 1 if pl.shields[name] else 0
        s[S_TROPHY + p] = -1 if pl.trophy is None else pl.trophy
        for key, n in pl.castle_taken.items():
            s[S_CASTLE + p * NL + E.LAND_INDEX[L.unkey(key)]] = n
        for pk in pl.wt_pairs:
            a, b = [int(gs[G_TSLOT + E.LAND_INDEX[L.unkey(x)]]) for x in pk.split("|")]
            s[S_PAIRS + p * MAXT + a] |= 1 << b
            s[S_PAIRS + p * MAXT + b] |= 1 << a
        for c, it in pl.fjord.items():
            i = E.FJORD_INDEX[c]
            k = it["kind"]
            if k in ("res", "bld"):
                s[S_FJ + p * NF + i] = E.SITE_ITEMS.index(it["type"]) + 1
            elif k == "ship":
                s[S_FJ + p * NF + i] = FJ_SHIP
                s[S_FJSHIP + p * NF + i] = it["id"]
                s[S_FJFLAG + p * NF + i] = 1 if it["filled"] else 0
            else:
                s[S_FJ + p * NF + i] = FJ_SITE
                s[S_FJFLAG + p * NF + i] = 1 if it["done"] else 0
    for i, owner in enumerate(g.trophy_owner):
        s[S_TOWNER + i] = -1 if owner is None else owner
    s[S_CUR] = g.current
    s[S_OVER] = 1 if g.game_over else 0
    s[S_PHASE] = PHASES[g.phase]
    s[S_PLACED] = g.placed_this_turn
    s[S_TOOK] = 1 if g.took_ship else 0
    return s, gs


# ---------------------------------------------------------------------------
# Rules (each one mirrors the method of looot.Game with the same name)
# ---------------------------------------------------------------------------

@njit(cache=True)
def is_anchor(s, gs, i):
    """Next to a Viking, or to an ocean space that counts."""
    for d in range(6):
        n = LAND_NB[i, d]
        if n == NL:
            continue
        if n >= NB:
            if s[S_OCEAN + n - NB] >= 0 or gs[G_RULES + R_LONG] == 0:
                return True
        elif s[S_VIK + n] + s[S_VIK + NL + n] > 0:
            return True
    return False


@njit(cache=True)
def legal_cells(s, gs, occupy, out):
    """Resource spaces where a Viking may go (occupied ones with occupy=True),
    in board order. Returns how many were written into `out`."""
    k = 0
    for i in range(NB):
        if gs[G_TERR + i] > 3:
            continue
        occ = s[S_VIK + i] + s[S_VIK + NL + i] > 0
        if occ != occupy:
            continue
        if is_anchor(s, gs, i):
            out[k] = i
            k += 1
    return k


@njit(cache=True)
def has_room(s, gs, p):
    buf = np.empty(NB, np.int64)
    if legal_cells(s, gs, False, buf) > 0:
        return True
    return s[S_SHIELD + p * 3 + 1] > 0 and legal_cells(s, gs, True, buf) > 0


@njit(cache=True)
def new_turn(s, gs):
    s[S_PLACED] = 0
    s[S_TOOK] = 0
    s[S_PHASE] = PH_PLACE
    p = s[S_CUR]
    if s[S_VLEFT + p] > 0 and not has_room(s, gs, p):
        s[S_VLEFT + p] -= 1           # nowhere to go: this Viking is lost
        s[S_PHASE] = PH_ACTIONS


@njit(cache=True)
def end_turn(s, gs):
    cur = s[S_CUR]
    for step in range(1, 3):
        nxt = (cur + step) % 2
        if s[S_VLEFT + nxt] > 0:
            s[S_CUR] = nxt
            new_turn(s, gs)
            return
    s[S_OVER] = 1
    s[S_PHASE] = PH_OVER


@njit(cache=True)
def fj_empty_count(s, p):
    n = 0
    for c in range(NF):
        if s[S_FJ + p * NF + c] == 0:
            n += 1
    return n


@njit(cache=True)
def adj_counts(s, p, c, cnt):
    """Items (7 types) on the fjord spaces next to fjord space c."""
    for t in range(7):
        cnt[t] = 0
    for d in range(6):
        n = FJ_NB[c, d]
        if n < NF:
            code = s[S_FJ + p * NF + n]
            if 1 <= code <= 7:
                cnt[code - 1] += 1


@njit(cache=True)
def need_of(s, gs, p, c, t):
    """How many items of type t the unfinished ship/site on fjord space c needs."""
    if s[S_FJ + p * NF + c] == FJ_SHIP:
        return LS_NEED[s[S_FJSHIP + p * NF + c], t] if t < 4 else 0
    return gs[G_SITENEED + (p * 3 + SITE_OF_CELL[c]) * 7 + t]


@njit(cache=True)
def update_fills(s, gs, p):
    """Flip longships and construction sites whose needs are all next to them."""
    cnt = np.zeros(7, np.int64)
    for c in range(NF):
        code = s[S_FJ + p * NF + c]
        if (code == FJ_SHIP or code == FJ_SITE) and s[S_FJFLAG + p * NF + c] == 0:
            adj_counts(s, p, c, cnt)
            ok = True
            for t in range(7):
                if need_of(s, gs, p, c, t) > cnt[t]:
                    ok = False
            if ok:
                s[S_FJFLAG + p * NF + c] = 1


@njit(cache=True)
def take(s, a, k):
    got = min(k, s[S_STOCK + a])
    s[S_STOCK + a] -= got
    return got


@njit(cache=True)
def tower_tiles(s, gs, p, towers, m, c, pay):
    """Towers that pay 1 tile (see Game.tower_tiles). towers: the m towers the
    chain touches, in (q, r) order. Writes them into `pay`, returns how many."""
    grp = np.full(m, -1, np.int64)
    stack = np.empty(m, np.int64)
    ng = 0
    for j in range(m):
        if grp[j] >= 0:
            continue
        grp[j] = ng
        top = 1
        stack[0] = j
        while top > 0:
            top -= 1
            a = stack[top]
            sa = gs[G_TSLOT + towers[a]]
            for b in range(m):
                if grp[b] < 0:
                    sb = gs[G_TSLOT + towers[b]]
                    if (s[S_PAIRS + p * MAXT + sa] >> sb) & 1:
                        grp[b] = ng
                        stack[top] = b
                        top += 1
        ng += 1
    # from now on all these towers count as linked to each other
    for a in range(m):
        sa = gs[G_TSLOT + towers[a]]
        for b in range(m):
            if a != b:
                s[S_PAIRS + p * MAXT + sa] |= 1 << gs[G_TSLOT + towers[b]]
    if ng < 2:
        return 0
    for gi in range(ng):
        best = -1
        bk0 = -1
        bk1 = -1
        bk2 = -1
        for j in range(m):
            if grp[j] != gi:
                continue
            a = towers[j]
            near = 0
            for d in range(6):
                if LAND_NB[c, d] == a:
                    near = 1
            k1 = np.int64(s[S_STOCK + a])
            k2 = COORD_RANK[a]
            # largest (next to the new Viking, tiles left, (q, r)), compared
            # in that order like Python compares tuples
            better = best < 0
            if not better:
                if near != bk0:
                    better = near > bk0
                elif k1 != bk1:
                    better = k1 > bk1
                else:
                    better = k2 > bk2
            if better:
                best = a
                bk0 = near
                bk1 = k1
                bk2 = k2
        pay[gi] = best
    return ng


@njit(cache=True)
def captures(s, gs, p, i, ptype, pfrom, n):
    """Buildings captured by player p's new Viking on i (see Game.captures).
    Appends one pending tile per tile taken; returns the new count."""
    if gs[G_RULES + R_BUILD] == 0:
        return n
    for d in range(6):                           # houses next to the Viking
        nb = LAND_NB[i, d]
        if nb < NB and gs[G_TERR + nb] == T_HOUSE:
            if take(s, nb, 1) == 1:
                ptype[n] = T_HOUSE
                pfrom[n] = nb
                n += 1
    # the chain of p's Vikings through i
    inchain = np.zeros(NL, np.bool_)
    stack = np.empty(NB, np.int64)
    inchain[i] = True
    top = 1
    stack[0] = i
    size = 1
    while top > 0:
        top -= 1
        c = stack[top]
        for d in range(6):
            nb = LAND_NB[c, d]
            if nb < NB and not inchain[nb] and s[S_VIK + p * NL + nb] > 0:
                inchain[nb] = True
                stack[top] = nb
                top += 1
                size += 1
    adj = np.zeros(NL, np.bool_)
    for c in range(NB):
        if inchain[c]:
            for d in range(6):
                nb = LAND_NB[c, d]
                if nb < NB:
                    t = gs[G_TERR + nb]
                    if t == T_TOWER or t == T_CASTLE:
                        adj[nb] = True
    towers = np.empty(MAXT, np.int64)
    m = 0
    for j in range(NB):
        a = COORD_ORDER[j]
        if adj[a] and gs[G_TERR + a] == T_TOWER:
            towers[m] = a
            m += 1
    pay = np.empty(MAXT, np.int64)
    npay = tower_tiles(s, gs, p, towers, m, i, pay)
    for j in range(npay):
        if take(s, pay[j], 1) == 1:
            ptype[n] = T_TOWER
            pfrom[n] = pay[j]
            n += 1
    level = min(3, size // 4)
    for j in range(NB):
        a = COORD_ORDER[j]
        if adj[a] and gs[G_TERR + a] == T_CASTLE:
            have = s[S_CASTLE + p * NL + a]
            if level > have:
                s[S_CASTLE + p * NL + a] = level
                got = take(s, a, level - have)
                for _ in range(got):
                    ptype[n] = T_CASTLE
                    pfrom[n] = a
                    n += 1
    return n


@njit(cache=True)
def drop_if_full(s, p, pfrom, n):
    """No empty fjord space left: pending tiles are lost (buildings go back)."""
    if n > 0 and fj_empty_count(s, p) == 0:
        for j in range(n):
            if pfrom[j] >= 0:
                s[S_STOCK + pfrom[j]] += 1
        return 0
    return n


@njit(cache=True)
def place_viking(s, gs, i, occupy, double, ptype, pfrom):
    """Place the current player's Viking on i. The tiles gained are written
    to ptype (item type) and pfrom (building space, -1 for a resource);
    returns how many are waiting to be placed on the fjord."""
    p = s[S_CUR]
    s[S_VIK + p * NL + i] += 1
    s[S_VLEFT + p] -= 1
    s[S_PLACED] += 1
    if occupy:
        s[S_SHIELD + p * 3 + 1] = 0
    if double:
        s[S_SHIELD + p * 3 + 2] = 0
    res = gs[G_TERR + i]
    n = 0
    ptype[n] = res
    pfrom[n] = -1
    n += 1
    if double:
        ptype[n] = res
        pfrom[n] = -1
        n += 1
    n = captures(s, gs, p, i, ptype, pfrom, n)
    n = drop_if_full(s, p, pfrom, n)
    s[S_PHASE] = PH_TILES if n > 0 else PH_ACTIONS
    return n


@njit(cache=True)
def place_tile(s, gs, ptype, pfrom, n, j, cell):
    """Put pending tile j on fjord space `cell`; returns the new pending count."""
    p = s[S_CUR]
    s[S_FJ + p * NF + cell] = ptype[j] + 1
    for k in range(j, n - 1):
        ptype[k] = ptype[k + 1]
        pfrom[k] = pfrom[k + 1]
    n -= 1
    update_fills(s, gs, p)
    n = drop_if_full(s, p, pfrom, n)
    if n == 0:
        s[S_PHASE] = PH_ACTIONS
    return n


@njit(cache=True)
def can_use_extra(s, gs):
    p = s[S_CUR]
    return (s[S_PHASE] == PH_ACTIONS and s[S_TOOK] == 0 and s[S_SHIELD + p * 3] > 0
            and s[S_VLEFT + p] > 0 and s[S_PLACED] > 0 and has_room(s, gs, p))


@njit(cache=True)
def use_extra(s):
    p = s[S_CUR]
    s[S_SHIELD + p * 3] = 0
    s[S_PHASE] = PH_PLACE


@njit(cache=True)
def take_ship(s, gs, slot, cell):
    """Take the longship in ocean slot `slot`, put it on fjord space `cell`,
    and refill the ocean from the top of the bag."""
    p = s[S_CUR]
    s[S_FJ + p * NF + cell] = FJ_SHIP
    s[S_FJSHIP + p * NF + cell] = s[S_OCEAN + slot]
    s[S_FJFLAG + p * NF + cell] = 0
    if s[S_BAGN] > 0:
        s[S_BAGN] -= 1
        s[S_OCEAN + slot] = s[S_BAG + s[S_BAGN]]
        s[S_BAG + s[S_BAGN]] = 0          # keep the unused part of the bag clean
    else:
        s[S_OCEAN + slot] = -1
    s[S_TOOK] = 1
    s[S_PHASE] = PH_ACTIONS
    update_fills(s, gs, p)


@njit(cache=True)
def axes(s, p):
    n = 0
    for c in range(NF):
        if s[S_FJ + p * NF + c] == 4:          # axe = item 3, code 4
            n += 1
    return n


@njit(cache=True)
def best_claimable(s, gs):
    """The highest trophy the player to move may claim now, or -1."""
    p = s[S_CUR]
    if s[S_TROPHY + p] >= 0 or s[S_PHASE] != PH_ACTIONS or gs[G_RULES + R_TROPHIES] == 0:
        return -1
    a = axes(s, p)
    best = -1
    for i in range(5):
        if s[S_TOWNER + i] < 0 and a >= T_AXES[i]:
            best = i
    return best


@njit(cache=True)
def claim(s, i):
    p = s[S_CUR]
    s[S_TOWNER + i] = p
    s[S_TROPHY + p] = i


@njit(cache=True)
def score_parts(s, p, count, value):
    """Fills count[6] / value[6] per score category; returns
    (construction site VP, unfilled longships, total score)."""
    for k in range(6):
        count[k] = 0
        value[k] = BASE[k]
    sites = 0
    unfilled = 0
    for c in range(NF):
        code = s[S_FJ + p * NF + c]
        if 1 <= code <= 7:
            for k in range(6):
                if CAT_ITEM[k] == code - 1:
                    count[k] += 1
        elif code == FJ_SHIP:
            sid = s[S_FJSHIP + p * NF + c]
            if s[S_FJFLAG + p * NF + c] > 0:
                value[LS_CAT[sid]] += LS_BONUS[sid]
            else:
                unfilled += 1
        elif code == FJ_SITE and s[S_FJFLAG + p * NF + c] > 0:
            sites += SITE_VP[SITE_OF_CELL[c]]
    total = 0
    for k in range(6):
        total += count[k] * value[k]
    tr = s[S_TROPHY + p]
    if tr >= 0:
        total += T_VP[tr]
    total += sites - 5 * unfilled
    return sites, unfilled, total


@njit(cache=True)
def total_score(s, p):
    count = np.zeros(6, np.int64)
    value = np.zeros(6, np.int64)
    return score_parts(s, p, count, value)[2]


@njit(cache=True)
def outcome(s, me):
    """(margin, win) for player `me` of a finished game (see Game.winners)."""
    a = total_score(s, me)
    b = total_score(s, 1 - me)
    ta = T_VP[s[S_TROPHY + me]] if s[S_TROPHY + me] >= 0 else -1
    tb = T_VP[s[S_TROPHY + 1 - me]] if s[S_TROPHY + 1 - me] >= 0 else -1
    if a > b or (a == b and ta > tb):
        win = 1.0
    elif a == b and ta == tb:
        win = 0.5
    else:
        win = 0.0
    return float(a - b), win


# ---------------------------------------------------------------------------
# Encoding: exactly the numbers of nn_encode.encode
# ---------------------------------------------------------------------------

@njit(cache=True)
def _chain_sizes(s, p, col, land):
    seen = np.zeros(NL, np.bool_)
    comp = np.empty(NB, np.int64)
    for start in range(NB):
        if seen[start] or s[S_VIK + p * NL + start] == 0:
            continue
        seen[start] = True
        comp[0] = start
        size = 1
        head = 0
        while head < size:
            c = comp[head]
            head += 1
            for d in range(6):
                nb = LAND_NB[c, d]
                if nb < NB and not seen[nb] and s[S_VIK + p * NL + nb] > 0:
                    seen[nb] = True
                    comp[size] = nb
                    size += 1
        v = min(size, 127)
        for j in range(size):
            land[comp[j], col] = v


@njit(cache=True)
def _encode_fjord(s, gs, p, out, glob, w):
    """nn_encode's fjord numbers for player p into out[NF, FJORD_F], and that
    player's score numbers into glob (w = 0 for me, 1 for the opponent)."""
    gain = s[S_VLEFT + p] + (1 if s[S_SHIELD + p * 3 + 2] > 0 else 0)
    cnt = np.zeros(7, np.int64)
    empty = 0
    spots = 0
    for c in range(NF):
        code = s[S_FJ + p * NF + c]
        room = 0
        hres = 0
        for d in range(6):
            n = FJ_NB[c, d]
            if n < NF:
                cn = s[S_FJ + p * NF + n]
                if cn == 0:
                    room += 1
                elif cn <= 4:
                    hres += 1
        out[c, FF_ROOM] = room
        if code == 0:
            empty += 1
            out[c, FF_EMPTY] = 1
            h = min(hres, 3)
            if h + room >= 3 and 3 - h <= gain:
                spots += 1
        elif code <= 4:
            out[c, FF_RES + code - 1] = 1
        elif code <= 7:
            out[c, FF_BLD + code - 5] = 1
        elif code == FJ_SHIP:
            sid = s[S_FJSHIP + p * NF + c]
            filled = s[S_FJFLAG + p * NF + c] > 0
            out[c, FF_SHIP_DONE if filled else FF_SHIP] = 1
            out[c, FF_SHIP_BONUS + LS_CAT[sid]] = LS_BONUS[sid]
            if not filled:
                adj_counts(s, p, c, cnt)
                nmiss = 0
                for r in range(4):
                    m = LS_NEED[sid, r] - cnt[r]
                    if m > 0:
                        out[c, FF_SHIP_MISSING + r] = m
                        nmiss += m
                out[c, FF_FILLABLE] = 1 if (nmiss <= room and nmiss <= gain) else 0
                out[c, FF_SLACK] = max(-13, min(13, gain - nmiss))
        else:
            k = SITE_OF_CELL[c]
            done = s[S_FJFLAG + p * NF + c] > 0
            out[c, FF_SITE_DONE if done else FF_SITE] = 1
            out[c, FF_SITE_VP] = SITE_VP[k]
            if not done:
                adj_counts(s, p, c, cnt)
                nmiss = 0
                nres = 0
                for t in range(7):
                    m = gs[G_SITENEED + (p * 3 + k) * 7 + t] - cnt[t]
                    if m > 0:
                        out[c, FF_SITE_MISSING + t] = m
                        nmiss += m
                        if t < 4:
                            nres += m
                out[c, FF_FILLABLE] = 1 if (nmiss <= room and nres <= gain) else 0
                out[c, FF_SLACK] = max(-13, min(13, gain - nmiss))
    count = np.zeros(6, np.int64)
    value = np.zeros(6, np.int64)
    sites, unfilled, total = score_parts(s, p, count, value)
    glob[GI_SPOTS[w]] = spots
    glob[GI_VIK[w]] = s[S_VLEFT + p]
    for k in range(3):
        glob[GI_SHIELD[w, k]] = 1 if s[S_SHIELD + p * 3 + k] > 0 else 0
    glob[GI_AXES[w]] = axes(s, p)
    for k in range(6):
        glob[GI_COUNT[w, k]] = count[k]
        glob[GI_VALUE[w, k]] = value[k]
    glob[GI_SITES[w]] = sites
    glob[GI_UNFILLED[w]] = unfilled
    glob[GI_TOTAL[w]] = total
    glob[GI_ROOM[w]] = empty


@njit(cache=True)
def encode(s, gs, me, land, fjord, glob):
    """Fill the ZEROED arrays land[NL, LAND_F], fjord[2, NF, FJORD_F] and
    glob[GLOB_F] with the numbers of nn_encode.encode for player `me`."""
    opp = 1 - me
    for i in range(NB):
        land[i, LF_TERRAIN + gs[G_TERR + i]] = 1
        land[i, LF_MY_VIK] = s[S_VIK + me * NL + i]
        land[i, LF_OPP_VIK] = s[S_VIK + opp * NL + i]
        land[i, LF_STOCK] = s[S_STOCK + i]
        land[i, LF_MY_CASTLE] = s[S_CASTLE + me * NL + i]
        land[i, LF_OPP_CASTLE] = s[S_CASTLE + opp * NL + i]
    for k in range(5):
        land[NB + k, LF_OCEAN] = 1
        if s[S_OCEAN + k] >= 0:
            land[NB + k, LF_SHIP] = 1
    _chain_sizes(s, me, LF_MY_CHAIN, land)
    _chain_sizes(s, opp, LF_OPP_CHAIN, land)
    # tower links, counted on the tower whose "q,r" text comes first
    nt = gs[G_NT]
    for w in range(2):
        p = me if w == 0 else opp
        col = LF_MY_TOWERS if w == 0 else LF_OPP_TOWERS
        for a in range(nt):
            ma = s[S_PAIRS + p * MAXT + a]
            for b in range(a + 1, nt):
                if (ma >> b) & 1:
                    ca = gs[G_TCELL + a]
                    cb = gs[G_TCELL + b]
                    first = ca if KEY_RANK[ca] < KEY_RANK[cb] else cb
                    land[first, col] += 1
    # free resource spaces where a Viking may be placed
    for i in range(NB):
        t = gs[G_TERR + i]
        if t <= 3 and s[S_VIK + i] + s[S_VIK + NL + i] == 0 and is_anchor(s, gs, i):
            land[i, LF_ANCHOR] = 1
            glob[GI_ANCHORS[t]] += 1
    _encode_fjord(s, gs, me, fjord[0], glob, 0)
    _encode_fjord(s, gs, opp, fjord[1], glob, 1)
    glob[GI_TOMOVE] = 1 if (s[S_CUR] == me and s[S_OVER] == 0) else 0
    for i in range(5):
        o = s[S_TOWNER + i]
        if o == me:
            glob[GI_TROPHY[i, 0]] = 1
        elif o == opp:
            glob[GI_TROPHY[i, 1]] = 1
    glob[GI_BAG] = s[S_BAGN]
    for k in range(5):
        sid = s[S_OCEAN + k]
        if sid >= 0:
            for r in range(4):
                glob[GI_OCEAN_RES[k, r]] += LS_NEED[sid, r]
            glob[GI_OCEAN_BONUS[k, LS_CAT[sid]]] = LS_BONUS[sid]


@njit(cache=True)
def encode_many(states, n, gs, me, land, fjord, glob):
    """Encode states[0:n] for player `me` into the first n rows of the arrays."""
    for k in range(n):
        land[k] = 0
        fjord[k] = 0
        glob[k] = 0
        encode(states[k], gs, me, land[k], fjord[k], glob[k])


def encode_state(s, gs, me):
    """nn_encode-style (land, fjord, glob) arrays for one state."""
    land = np.zeros((NL, LAND_F), np.int8)
    fjord = np.zeros((2, NF, FJORD_F), np.int8)
    glob = np.zeros(GLOB_F, np.int16)
    encode(s, gs, me, land, fjord, glob)
    return land, fjord, glob


# ---------------------------------------------------------------------------
# The network player's moves (mirrors nn_bot.NNBot, whose heuristics come
# from looot.Bot: where tiles and longships go on the fjord)
# ---------------------------------------------------------------------------

@njit(cache=True)
def seed(x):
    np.random.seed(x)


@njit(cache=True)
def placement_options(s, gs, cells, occs, dbls):
    """Like Bot.placement_options: every legal space, then again with the
    double shield, then the occupied spaces with the occupy shield."""
    buf = np.empty(NB, np.int64)
    n0 = legal_cells(s, gs, False, buf)
    k = 0
    for j in range(n0):
        cells[k] = buf[j]
        occs[k] = 0
        dbls[k] = 0
        k += 1
    p = s[S_CUR]
    if s[S_SHIELD + p * 3 + 2] > 0:
        for j in range(n0):
            cells[k] = buf[j]
            occs[k] = 0
            dbls[k] = 1
            k += 1
    if s[S_SHIELD + p * 3 + 1] > 0:
        n1 = legal_cells(s, gs, True, buf)
        for j in range(n1):
            cells[k] = buf[j]
            occs[k] = 1
            dbls[k] = 0
            k += 1
    return k


@njit(cache=True)
def best_tile_cell(s, gs, p, t, bw, noise):
    """Bot.best_tile_cell: an empty fjord space next to a longship/site that
    still needs this tile, away from ones that don't, near open space."""
    cnt = np.zeros(7, np.int64)
    best = -1
    best_v = 0.0
    for c in range(NF):
        if s[S_FJ + p * NF + c] != 0:
            continue
        v = 0.0
        for d in range(6):
            n = FJ_NB[c, d]
            if n == NF:
                continue
            code = s[S_FJ + p * NF + n]
            if (code == FJ_SHIP or code == FJ_SITE) and s[S_FJFLAG + p * NF + n] == 0:
                adj_counts(s, p, n, cnt)
                if need_of(s, gs, p, n, t) > cnt[t]:
                    v += bw[0]
                else:
                    v -= bw[1]
            elif code == 0:
                v += bw[2]
        if noise:
            v += np.random.random() * 0.1
        if best < 0 or v > best_v:
            best = c
            best_v = v
    return best


@njit(cache=True)
def best_ship_cell(s, p, sid, bw, noise):
    """Bot.best_ship_cell: next to the resources it needs, with room left."""
    cnt = np.zeros(7, np.int64)
    best = -1
    best_v = 0.0
    for c in range(NF):
        if s[S_FJ + p * NF + c] != 0:
            continue
        adj_counts(s, p, c, cnt)
        m = 0
        for r in range(4):
            m += min(LS_NEED[sid, r], cnt[r])
        room = 0
        for d in range(6):
            n = FJ_NB[c, d]
            if n < NF and s[S_FJ + p * NF + n] == 0:
                room += 1
        v = m * bw[3] + min(room, 3 - m) * bw[4]
        if noise:
            v += np.random.random() * 0.1
        if room < 3 - m:
            v -= 20
        if best < 0 or v > best_v:
            best = c
            best_v = v
    return best


@njit(cache=True)
def place_with_tiles(s, gs, cell, occ, dbl, bw, noise, ptype, pfrom):
    """Place a Viking and put all its tiles on the fjord the greedy way."""
    n = place_viking(s, gs, cell, occ, dbl, ptype, pfrom)
    p = s[S_CUR]
    while n > 0:
        j = 0
        if n > fj_empty_count(s, p):          # not everything fits: best first
            best = -1.0
            for k in range(n):
                if KEEP[ptype[k]] > best:
                    best = KEEP[ptype[k]]
                    j = k
        c = best_tile_cell(s, gs, p, ptype[j], bw, noise)
        n = place_tile(s, gs, ptype, pfrom, n, j, c)


@njit(cache=True)
def firsts(s, gs, bw, noise, placed, plain):
    """Every Viking placement of the player to move: `placed` gets the state
    after placing (and putting the tiles away), `plain` the same state with
    the turn simply ended (for a quick first judgement). Returns how many."""
    cells = np.empty(MAXO, np.int64)
    occs = np.empty(MAXO, np.int64)
    dbls = np.empty(MAXO, np.int64)
    n = placement_options(s, gs, cells, occs, dbls)
    ptype = np.empty(MAXP, np.int64)
    pfrom = np.empty(MAXP, np.int64)
    for k in range(n):
        placed[k] = s
        place_with_tiles(placed[k], gs, cells[k], occs[k] == 1, dbls[k] == 1,
                         bw, noise, ptype, pfrom)
        plain[k] = placed[k]
        end_turn(plain[k], gs)
    return n


@njit(cache=True)
def endings(st, gs, bw, noise, shuffle, out, k):
    """NNBot._endings: every way to finish the turn from st - no longship or
    one of the ocean's, then no trophy or the best one - with the turn
    ended. Written into out[k:]; returns the new k."""
    p = st[S_CUR]
    slots = np.full(6, -1, np.int64)
    ns = 1
    if st[S_TOOK] == 0 and fj_empty_count(st, p) > 0:
        for slot in range(5):
            if st[S_OCEAN + slot] >= 0:
                slots[ns] = slot
                ns += 1
    for j in range(ns):
        slot = slots[j]
        g3 = st.copy()
        if slot >= 0:
            if shuffle:
                # the copy's bag is in the real order; a player can't know
                # which longship comes out next, so shuffle it
                nb = g3[S_BAGN]
                for a in range(nb - 1, 0, -1):
                    b = np.random.randint(0, a + 1)
                    tmp = g3[S_BAG + a]
                    g3[S_BAG + a] = g3[S_BAG + b]
                    g3[S_BAG + b] = tmp
            cell = best_ship_cell(g3, p, g3[S_OCEAN + slot], bw, noise)
            take_ship(g3, gs, slot, cell)
        tr = best_claimable(g3, gs)
        for option in range(2 if tr >= 0 else 1):
            out[k] = g3
            if option == 1:
                claim(out[k], tr)
            end_turn(out[k], gs)
            k += 1
    return k


def judge_states(net, requests):
    """The network's predicted final margin for many batches of states at
    once. requests: list of (states, n, gs, me). Finished games count
    exactly; all other states go through the network in ONE call.
    Returns one array of values per request."""
    rows = []
    vals = []
    for states, n, gs, me in requests:
        v = np.empty(n)
        over = states[:n, S_OVER] == 1
        for k in np.where(over)[0]:
            v[k] = outcome(states[k], me)[0]
        todo = np.where(~over)[0]
        rows.append((states, gs, me, todo))
        vals.append(v)
    total = sum(len(r[3]) for r in rows)
    if total:
        land = np.zeros((total, NL, LAND_F), np.int8)
        fjord = np.zeros((total, 2, NF, FJORD_F), np.int8)
        glob = np.zeros((total, GLOB_F), np.int16)
        at = 0
        for states, gs, me, todo in rows:
            m = len(todo)
            if m:
                encode_many(np.ascontiguousarray(states[todo]), m, gs, me,
                            land[at:at + m], fjord[at:at + m], glob[at:at + m])
                at += m
        margin = net(land, fjord, glob)[0]
        at = 0
        for (states, gs, me, todo), v in zip(rows, vals):
            m = len(todo)
            v[todo] = margin[at:at + m]
            at += m
    return vals


class FastPlayer:
    """The network player (same choices as nn_bot.NNBot) on array states."""

    TOP_FULL = 4            # placements worked out with every ending
    TOP_EXTRA = 2           # second placements (shield) worked out in full

    def __init__(self, net, rng, explore=0.0, noise=True, shuffle=True, weights=None):
        self.net = net
        self.rng = rng
        self.explore = explore
        self.noise = noise
        self.shuffle = shuffle
        self.bw = bot_weights(weights)
        # placed / plain states of the Viking placements, the same for a
        # second Viking (shield), and the finished-turn candidates
        self.buf = [np.empty((n, S_LEN), np.int32) for n in (MAXO, MAXO, MAXO, MAXO, 512)]

    def turn_steps(self, s, gs):
        """One turn as a generator. Whenever it needs the network, it yields
        (states, n, me) and expects the values of states[0:n] back through
        send(). It ends by returning the state after the turn.

        Written this way, one implementation serves both: turn() below
        answers each question at once, and selfplay_batched() collects the
        questions of many games to ask the network (on the GPU) together."""
        me = int(s[S_CUR])
        placed, plain, placed2, plain2, out = self.buf
        if s[S_PHASE] == PH_PLACE:
            n = firsts(s, gs, self.bw, self.noise, placed, plain)
            if n <= self.TOP_FULL:
                idx = np.arange(n)
            else:
                vals = yield (plain, n, me)
                idx = np.argsort(-vals)[:self.TOP_FULL]
            chosen = [placed[i].copy() for i in idx]
        else:
            chosen = [s.copy()]
        k = 0
        for st in chosen:
            k = endings(st, gs, self.bw, self.noise, self.shuffle, out, k)
            if s[S_PHASE] == PH_PLACE and can_use_extra(st, gs):
                sx = st.copy()
                use_extra(sx)
                n2 = firsts(sx, gs, self.bw, self.noise, placed2, plain2)
                if n2 <= self.TOP_EXTRA:
                    idx2 = np.arange(n2)
                else:
                    vals = yield (plain2, n2, me)
                    idx2 = np.argsort(-vals)[:self.TOP_EXTRA]
                for i in idx2:
                    k = endings(placed2[i].copy(), gs, self.bw, self.noise, self.shuffle, out, k)
        if k == 0:
            s2 = s.copy()
            end_turn(s2, gs)
            return s2
        vals = yield (out, k, me)
        if self.explore and self.rng.random() < self.explore:
            pick = self.rng.randrange(k)
        elif self.noise:
            pick = int(np.argmax(vals + np.array([self.rng.random() * 1e-3 for _ in range(k)])))
        else:
            pick = int(np.argmax(vals))
        return out[pick].copy()

    def turn(self, s, gs):
        """The state after the player to move plays its turn."""
        steps = self.turn_steps(s, gs)
        try:
            req = next(steps)
            while True:
                states, n, me = req
                req = steps.send(judge_states(self.net, [(states, n, gs, me)])[0])
        except StopIteration as done:
            return done.value


def record(states, gs, seed_):
    """Training data of a finished game: every turn-start position seen by
    both players, labelled with how the game ended for that player. Rows:
    turn 0 (player 0, player 1), turn 1 (player 0, player 1), ... - the
    format of gen_data.play_and_record."""
    n = len(states)
    arr = np.stack(states)
    land = np.zeros((2 * n, NL, LAND_F), np.int8)
    fjord = np.zeros((2 * n, 2, NF, FJORD_F), np.int8)
    glob = np.zeros((2 * n, GLOB_F), np.int16)
    for me in range(2):
        l, f, gl = land[me::2], fjord[me::2], glob[me::2]
        encode_many(arr, n, gs, me, l, f, gl)
    res = [outcome(states[-1], me) for me in range(2)]
    margin = np.array([res[k % 2][0] for k in range(2 * n)], np.float32)
    win = np.array([res[k % 2][1] for k in range(2 * n)], np.float32)
    game = np.full(2 * n, seed_, np.int64)
    return land, fjord, glob, margin, win, game


def selfplay_game(seed_, layout, rules, net, explore=0.0, noise=True):
    """One game between two network players, played on its own. Returns
    (land, fjord, glob, margin, win, game) like gen_data.play_and_record."""
    import random
    g = L.Game([("p0", True), ("p1", True)], seed_, layout, L.rules_for(rules))
    s, gs = from_game(g)
    seed(seed_ % (2 ** 31))
    player = FastPlayer(net, random.Random(seed_ * 7), explore, noise=noise, shuffle=noise)
    states = [s]
    while s[S_OVER] == 0:
        s = player.turn(s, gs)
        states.append(s)
        assert len(states) < 200
    return record(states, gs, seed_)


def selfplay_batched(seeds, layout, rules, net, explore=0.0, parallel=64, noise=True):
    """Self-play of many games at once, in step: every game runs until its
    player needs the network, then the positions of ALL games are judged in
    one batch (on the GPU with TorchNet), and every game gets its answers.
    Yields the training data of each game as it finishes (see record())."""
    import random
    todo = list(seeds)
    seed((todo[0] if todo else 0) % (2 ** 31))
    slots = []                  # one per game being played

    def start(sd, player=None):
        g = L.Game([("p0", True), ("p1", True)], sd, layout, L.rules_for(rules))
        s, gs = from_game(g)
        rng = random.Random(sd * 7)
        if player is None:
            player = FastPlayer(net, rng, explore, noise=noise, shuffle=noise)
        player.rng = rng
        return {"seed": sd, "s": s, "gs": gs, "player": player, "states": [s],
                "steps": None, "req": None}

    def advance(slot, answer):
        """Run the game until its next question, or until it ends. Returns
        the finished game's data, or None."""
        while True:
            try:
                if slot["steps"] is None:
                    slot["steps"] = slot["player"].turn_steps(slot["s"], slot["gs"])
                    slot["req"] = next(slot["steps"])
                else:
                    slot["req"] = slot["steps"].send(answer)
                return None
            except StopIteration as done:
                slot["s"] = done.value
                slot["states"].append(done.value)
                slot["steps"] = None
                slot["req"] = None
                answer = None
                if done.value[S_OVER]:
                    return record(slot["states"], slot["gs"], slot["seed"])

    while todo or slots:
        while todo and len(slots) < parallel:
            slot = start(todo.pop(0))
            data = advance(slot, None)
            if data is not None:               # (a game can't end without a question)
                yield data
                continue
            slots.append(slot)
        answers = judge_states(net, [(sl["req"][0], sl["req"][1], sl["gs"], sl["req"][2])
                                     for sl in slots])
        keep = []
        for slot, ans in zip(slots, answers):
            data = advance(slot, ans)
            if data is None:
                keep.append(slot)
                continue
            yield data
            if todo:                            # reuse the player (and its buffers)
                nxt = start(todo.pop(0), slot["player"])
                d2 = advance(nxt, None)
                if d2 is None:
                    keep.append(nxt)
                else:
                    yield d2
        slots = keep


class TorchNet:
    """The value network in PyTorch, on the GPU: the same numbers as
    nn_bot.FastNet (up to rounding), but fast for batches of thousands."""

    CHUNK = 4096            # positions per step on the GPU (limits memory)

    def __init__(self, name, device=None):
        import torch
        from nn_model import ValueNet
        path = name if name.endswith(".pt") else os.path.join(HERE, "models", name + ".pt")
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ValueNet()
        self.model.load_state_dict(torch.load(path, map_location="cpu")["state"])
        self.model.eval().to(self.dev)

    @staticmethod
    def _unique(x):
        """Rows of x without repeats of the row just before, and for every row
        which of those it is. The candidate turns come in groups (all endings
        of one Viking placement, all turns of one game) whose boards repeat
        in a row, so comparing each row with the previous one finds almost
        every duplicate - with one quick comparison instead of sorting."""
        flat = x.reshape(len(x), -1)
        new = np.empty(len(x), np.bool_)
        new[0] = True
        new[1:] = (flat[1:] != flat[:-1]).any(1)
        return x[new], np.cumsum(new) - 1

    def __call__(self, land, fjord, glob):
        """Many positions share boards: the opponent's fjord is the same in
        all candidate turns of a game, the landscape the same for all
        endings of one Viking placement. Each board part of the network runs
        once per DIFFERENT board; the results are then put back in place."""
        torch = self.torch
        F_ = torch.nn.functional
        mdl = self.model
        ms, ws = [], []
        with torch.no_grad():
            for i in range(0, len(land), self.CHUNK):
                part = slice(i, i + self.CHUNK)
                outs = []
                for x, board, mu, sd in ((land[part], mdl.land, mdl.land_mu, mdl.land_sd),
                                         (fjord[part, 0], mdl.fjord, mdl.fjord_mu, mdl.fjord_sd),
                                         (fjord[part, 1], mdl.fjord, mdl.fjord_mu, mdl.fjord_sd)):
                    u, inv = self._unique(x)
                    xu = (torch.from_numpy(u).to(self.dev).float() - mu) / sd
                    outs.append(board(xu)[torch.from_numpy(inv).to(self.dev)])
                gl = (torch.from_numpy(glob[part]).to(self.dev).float() - mdl.glob_mu) / mdl.glob_sd
                d = F_.relu(mdl.glob(gl))
                h = F_.relu(mdl.fc1(torch.cat(outs + [d], dim=1)))
                h = F_.relu(mdl.fc2(h))
                ms.append(mdl.margin(h).squeeze(1) * 20.0)
                ws.append(torch.sigmoid(mdl.win(h).squeeze(1)))
        return torch.cat(ms).cpu().numpy(), torch.cat(ws).cpu().numpy()
