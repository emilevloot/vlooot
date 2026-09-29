"""
Looot for 2, 3 or 4 players as numpy arrays, compiled with Numba: the
multi-player copy of fastgame.py (same rules, same network player), with
the encoding of mp_encode.py. Up to 4 players in one state array: every
per-player block has room for MAXPL = 4; gs[G_NP] says how many play.
Spaces of landscape boards that are not in the game have terrain T_OUT.

test_mp.py checks that it plays exactly like looot.py.
"""

import glob
import hashlib
import os

import numpy as np
from numba import njit

import looot as L
import mp_encode as E

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
MAXO = 400           # placement options in one turn

# terrain codes = index in E.TERRAINS; the ocean gets 7
assert E.TERRAINS == ["forest", "field", "mountain", "battlefield",
                      "house", "watchtower", "castle"]
T_HOUSE, T_TOWER, T_CASTLE, T_OCEAN, T_OUT = 4, 5, 6, 7, 8   # T_OUT: board not in this game
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
# terrain codes 0-3 give the resources in E.RES order
assert [L.TERRAIN_RESOURCE[t] for t in E.TERRAINS[:4]] == E.RES

# ---------------------------------------------------------------------------
# State layout: offsets into the int32 state array `s`
# ---------------------------------------------------------------------------

MAXPL = E.MAX_PLAYERS
S_VIK = 0                       # [MAXPL][NL] Vikings per player per space
S_STOCK = S_VIK + MAXPL * NL    # [NL] tiles left on a building
S_OCEAN = S_STOCK + NL          # [5] longship id, or -1
S_BAG = S_OCEAN + 5             # [30] the bag; the top is S_BAG + bag size - 1
S_BAGN = S_BAG + 30
S_VLEFT = S_BAGN + 1            # [MAXPL] Vikings left
S_SHIELD = S_VLEFT + MAXPL      # [MAXPL][3] extra, occupy, double
S_TROPHY = S_SHIELD + 3 * MAXPL # [MAXPL] trophy index, or -1
S_TOWNER = S_TROPHY + MAXPL     # [5] owner of each trophy, or -1
S_CASTLE = S_TOWNER + 5         # [MAXPL][NL] castle tiles taken per castle space
S_PAIRS = S_CASTLE + MAXPL * NL # [MAXPL][MAXT] bitmask of linked towers per tower
S_FJ = S_PAIRS + MAXPL * MAXT   # [MAXPL][NF] fjord codes
S_FJSHIP = S_FJ + MAXPL * NF    # [MAXPL][NF] longship id on a fjord space
S_FJFLAG = S_FJSHIP + MAXPL * NF  # [MAXPL][NF] longship filled / site done
S_CUR = S_FJFLAG + MAXPL * NF   # player to move
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
G_SITENEED = G_NT + 1           # [MAXPL][3][7] what each construction site needs
G_RULES = G_SITENEED + MAXPL * 3 * 7   # [5] buildings, sites, longships, shields, trophies
G_NP = G_RULES + 5              # number of players
G_LEN = G_NP + 1
R_BUILD, R_SITES, R_LONG, R_SHIELDS, R_TROPHIES = 0, 1, 2, 3, 4
RULE_KEYS = ["buildings", "sites", "longships", "shields", "trophies"]

# bot weights (passed in as an array, so changed weights need no recompile)
BW_KEYS = ["tile_need", "tile_block", "tile_open", "shipcell_match", "shipcell_room"]
KEEP = np.array([1.0, 1.0, 2.0, 1.5, 2.0, 3.0, 5.0])   # Bot.KEEP_ORDER per item type


def bot_weights(w=None):
    w = dict(L.BOT_WEIGHTS, **(w or {}))
    return np.array([w[k] for k in BW_KEYS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Encoding layout (the same numbers as nn_encode.encode_reference)
# ---------------------------------------------------------------------------

LAND_F, FJORD_F, GLOB_F = E.LAND_F, E.FJORD_F, E.GLOB_F
LF_TERRAIN, LF_OCEAN, LF_SHIP, LF_VIK = E.LF_TERRAIN, E.LF_OCEAN, E.LF_SHIP, E.LF_VIK
LF_FREE, LF_STACK, LF_CHAIN, LF_STOCK = E.LF_FREE, E.LF_STACK, E.LF_CHAIN, E.LF_STOCK
LF_TLINKS, LF_TTOUCH, LF_CLEVEL, LF_CNEXT = E.LF_TLINKS, E.LF_TTOUCH, E.LF_CLEVEL, E.LF_CNEXT
LF_GAIN, N_GAIN, LF_TURNS, LF_INPLAY = E.LF_GAIN, len(E.GAIN), E.LF_TURNS, E.LF_INPLAY
FF_EMPTY, FF_RES, FF_BLD = E.FF_EMPTY, E.FF_RES, E.FF_BLD
FF_SHIP, FF_SHIP_DONE, FF_SHIP_MISSING, FF_SHIP_BONUS = (
    E.FF_SHIP, E.FF_SHIP_DONE, E.FF_SHIP_MISSING, E.FF_SHIP_BONUS)
FF_SITE, FF_SITE_DONE, FF_SITE_MISSING, FF_SITE_VP = (
    E.FF_SITE, E.FF_SITE_DONE, E.FF_SITE_MISSING, E.FF_SITE_VP)
FF_ROOM, FF_FILLABLE, FF_SLACK, FF_TURNS = E.FF_ROOM, E.FF_FILLABLE, E.FF_SLACK, E.FF_TURNS

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
GI_NEED = np.array([[G_["need_%s_%s" % (t, w)] for t in E.SITE_ITEMS] for w in ("me", "opp")])
GI_BEST_GAIN = np.array([G_["best_gain_me"], G_["best_gain_opp"]])
GI_BEST_USEFUL = np.array([G_["best_useful_me"], G_["best_useful_opp"]])
GI_TURNS = np.array([G_["turns_left_me"], G_["turns_left_opp"]])
GI_TURNS_HOT = np.array([[G_["turns_%s_%d" % (w, t)] for t in range(E.MAX_TURNS + 1)]
                         for w in ("me", "opp")])
GI_TOMOVE, GI_BAG = G_["to_move_me"], G_["bag"]
GI_PLAYERS = np.array([G_["players_%d" % n] for n in (2, 3, 4)])
GI_SEAT = np.array([[G_["%s_%s" % (w, s_)] for w in ("present", "total", "vik", "turns", "strongest")]
                    for s_ in E.SEATS])


def _check_cache():
    """Numba keeps compiled code on disk. It notices when THIS file changes,
    but not when nn_encode.py or looot.py change numbers used here, so we
    fingerprint those numbers and clear the compiled code when they differ."""
    sig = hashlib.sha1(repr((E.GLOB_NAMES, E.LAND_F, E.FJORD_F, E.ENCODING_VERSION,
                             L.LONGSHIPS, L.TROPHIES, L.BASE_VALUE, L.SITE_VP,
                             L.FJORD_SITE_SPOTS, L.OCEAN_CELLS, L.BOARD_PLACEMENT,
                             L.BOARD_SHAPE, sorted(L.FJORD_COLUMNS.items()))).encode()).hexdigest()
    cache = os.path.join(HERE, "__pycache__")
    stamp = os.path.join(cache, "mp_game.sig")
    old = open(stamp).read() if os.path.exists(stamp) else None
    if old != sig:
        for f in glob.glob(os.path.join(cache, "mp_game.*.nb[ic]")):
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
    assert 2 <= len(g.players) <= MAXPL
    s = np.zeros(S_LEN, dtype=np.int32)
    gs = np.zeros(G_LEN, dtype=np.int64)
    gs[G_TERR:G_TERR + NL] = T_OCEAN
    gs[G_TSLOT:G_TSLOT + NL] = -1
    gs[G_TCELL:G_TCELL + MAXT] = -1
    nt = 0
    gs[G_NP] = len(g.players)
    for i in range(NB):
        cell = E.LAND_CELLS[i]
        if cell not in g.land:
            gs[G_TERR + i] = T_OUT
            continue
        code = E.TERRAINS.index(g.land[cell])
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
        for p in range(len(g.players)):
            s[S_VIK + p * NL + i] = v.count(p)
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
def occupied(s, gs, i):
    """Any player's Viking on space i."""
    for p in range(gs[G_NP]):
        if s[S_VIK + p * NL + i] > 0:
            return True
    return False


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
        elif occupied(s, gs, n):
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
        occ = occupied(s, gs, i)
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
    npl = gs[G_NP]
    for step in range(1, npl + 1):
        nxt = (cur + step) % npl
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
def tower_groups(s, gs, p, towers, m, grp):
    """Split the m towers into groups of towers player p linked before:
    grp[j] = group of towers[j]. Returns the number of groups."""
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
    return ng


@njit(cache=True)
def tower_choice(s, towers, m, grp, ng, c, pay):
    """The tower of each group that pays: the one next to the new Viking on
    c, otherwise the one with most tiles left (then the highest (q, r))."""
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


@njit(cache=True)
def tower_tiles(s, gs, p, towers, m, c, pay):
    """Towers that pay 1 tile (see Game.tower_tiles). towers: the m towers the
    chain touches, in (q, r) order. Writes them into `pay`, returns how many."""
    grp = np.full(m, -1, np.int64)
    ng = tower_groups(s, gs, p, towers, m, grp)
    # from now on all these towers count as linked to each other
    for a in range(m):
        sa = gs[G_TSLOT + towers[a]]
        for b in range(m):
            if a != b:
                s[S_PAIRS + p * MAXT + sa] |= 1 << gs[G_TSLOT + towers[b]]
    if ng < 2:
        return 0
    tower_choice(s, towers, m, grp, ng, c, pay)
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
def outcome(s, gs, me):
    """(margin, win) for player `me` of a finished game (see Game.winners):
    my score minus the best other score; my share of the win."""
    npl = gs[G_NP]
    best_other = -1000000
    best_s = -1000000
    best_t = -2
    for p in range(npl):
        sc = total_score(s, p)
        tv = T_VP[s[S_TROPHY + p]] if s[S_TROPHY + p] >= 0 else -1
        if p != me:
            best_other = max(best_other, sc)
        if sc > best_s or (sc == best_s and tv > best_t):
            best_s = sc
            best_t = tv
    n_win = 0
    me_wins = False
    for p in range(npl):
        sc = total_score(s, p)
        tv = T_VP[s[S_TROPHY + p]] if s[S_TROPHY + p] >= 0 else -1
        if sc == best_s and tv == best_t:
            n_win += 1
            if p == me:
                me_wins = True
    return float(total_score(s, me) - best_other), (1.0 / n_win) if me_wins else 0.0


# ---------------------------------------------------------------------------
# Encoding: exactly the numbers of mp_encode.encode_reference
# ---------------------------------------------------------------------------

@njit(cache=True)
def _chains(s, gs, p, comp, size, adj):
    """Label the chains of player p's Vikings: comp[i] = chain number of
    board space i (-1: none of p's Vikings), size[k] = its number of spaces,
    adj[k] = bitmask (bit = board space) of the watchtowers and castles next
    to it (two int64 words: spaces 0-63 and 64-127). Returns the number of chains."""
    for i in range(NL):
        comp[i] = -1
    todo = np.empty(NB, np.int64)
    nc = 0
    for start in range(NB):
        if comp[start] >= 0 or s[S_VIK + p * NL + start] == 0:
            continue
        comp[start] = nc
        todo[0] = start
        n = 1
        head = 0
        m0 = np.int64(0)
        m1 = np.int64(0)
        while head < n:
            c = todo[head]
            head += 1
            for d in range(6):
                nb = LAND_NB[c, d]
                if nb >= NB:
                    continue
                t = gs[G_TERR + nb]
                if t == T_TOWER or t == T_CASTLE:
                    if nb < 64:
                        m0 |= np.int64(1) << nb
                    else:
                        m1 |= np.int64(1) << (nb - 64)
                if comp[nb] < 0 and s[S_VIK + p * NL + nb] > 0:
                    comp[nb] = nc
                    todo[n] = nb
                    n += 1
        size[nc] = n
        adj[nc, 0] = m0
        adj[nc, 1] = m1
        nc += 1
    return nc


@njit(cache=True)
def _bit(m0, m1, a):
    if a < 64:
        return (m0 >> a) & 1
    return (m1 >> (a - 64)) & 1


@njit(cache=True)
def _placement_gain(s, gs, p, c, comp, size, adj, need, out):
    """What player p would get from a Viking on free space c (no shields),
    without changing the game: out = [chain size, house, watchtower and
    castle tiles, useful items] (see nn_encode._placement_gain)."""
    seen = np.full(6, -1, np.int64)
    ns = 0
    total = 1
    m0 = np.int64(0)
    m1 = np.int64(0)
    for d in range(6):
        nb = LAND_NB[c, d]
        if nb >= NB:
            continue
        t = gs[G_TERR + nb]
        if t == T_TOWER or t == T_CASTLE:
            if nb < 64:
                m0 |= np.int64(1) << nb
            else:
                m1 |= np.int64(1) << (nb - 64)
        k = comp[nb]
        if k >= 0:
            new = True
            for j in range(ns):
                if seen[j] == k:
                    new = False
            if new:
                seen[ns] = k
                ns += 1
                total += size[k]
                m0 |= adj[k, 0]
                m1 |= adj[k, 1]
    houses = 0
    towers_got = 0
    castles = 0
    if gs[G_RULES + R_BUILD] > 0:
        for d in range(6):
            nb = LAND_NB[c, d]
            if nb < NB and gs[G_TERR + nb] == T_HOUSE and s[S_STOCK + nb] > 0:
                houses += 1
        towers = np.empty(MAXT, np.int64)
        m = 0
        for j in range(NB):
            a = COORD_ORDER[j]
            if _bit(m0, m1, a) and gs[G_TERR + a] == T_TOWER:
                towers[m] = a
                m += 1
        grp = np.full(m, -1, np.int64)
        ng = tower_groups(s, gs, p, towers, m, grp)
        if ng >= 2:
            pay = np.empty(MAXT, np.int64)
            tower_choice(s, towers, m, grp, ng, c, pay)
            for j in range(ng):
                if s[S_STOCK + pay[j]] > 0:
                    towers_got += 1
        level = min(3, total // 4)
        for a in range(NB):
            if _bit(m0, m1, a) and gs[G_TERR + a] == T_CASTLE:
                have = s[S_CASTLE + p * NL + a]
                if level > have:
                    castles += min(level - have, s[S_STOCK + a])
    out[0] = total
    out[1] = houses
    out[2] = towers_got
    out[3] = castles
    out[4] = (min(1, need[gs[G_TERR + c]]) + min(houses, need[4])
              + min(towers_got, need[5]) + min(castles, need[6]))


@njit(cache=True)
def _encode_fjord(s, gs, p, out, need):
    """nn_encode's fjord numbers for player p into out[NF, FJORD_F] (all but
    the clock), and the items still missing on ships/sites that can be
    finished into need[7]. Returns the number of empty spaces where a new
    longship could still be filled."""
    gain = s[S_VLEFT + p] + (1 if s[S_SHIELD + p * 3 + 2] > 0 else 0)
    cnt = np.zeros(7, np.int64)
    spots = 0
    for t in range(7):
        need[t] = 0
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
                ok = nmiss <= room and nmiss <= gain
                out[c, FF_FILLABLE] = 1 if ok else 0
                out[c, FF_SLACK] = max(-13, min(13, gain - nmiss))
                if ok:
                    for r in range(4):
                        need[r] += out[c, FF_SHIP_MISSING + r]
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
                ok = nmiss <= room and nres <= gain
                out[c, FF_FILLABLE] = 1 if ok else 0
                out[c, FF_SLACK] = max(-13, min(13, gain - nmiss))
                if ok:
                    for t in range(7):
                        need[t] += out[c, FF_SITE_MISSING + t]
    return spots


@njit(cache=True)
def _glob_player(s, p, glob, w, spots, need, tl):
    """The global numbers of player p: w = 0 as "me", 1 as "opp"."""
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
    glob[GI_ROOM[w]] = fj_empty_count(s, p)
    for t in range(7):
        glob[GI_NEED[w, t]] = need[t]
    glob[GI_TURNS[w]] = tl
    glob[GI_TURNS_HOT[w, tl]] = 1


@njit(cache=True)
def turns_left(s, p):
    """Turns player p still gets (the +1 Viking shield puts 2 in one turn)."""
    v = s[S_VLEFT + p]
    return v - 1 if (s[S_SHIELD + p * 3] > 0 and v >= 2) else v


@njit(cache=True)
def encode(s, gs, me, land, fjord, glob):
    """Fill the ZEROED arrays land[NL, LAND_F], fjord[MAXPL, NF, FJORD_F] and
    glob[GLOB_F] with the numbers of mp_encode.encode_reference for `me`."""
    npl = gs[G_NP]
    build = gs[G_RULES + R_BUILD] > 0
    seat = np.empty(npl, np.int64)
    for k in range(npl):
        seat[k] = (me + k) % npl
    tl = np.zeros(MAXPL, np.int64)
    for p in range(npl):
        tl[p] = turns_left(s, p)
    # fjords of every seat, shopping lists
    need = np.zeros((MAXPL, 7), np.int64)
    spots = np.zeros(MAXPL, np.int64)
    for k in range(npl):
        p = seat[k]
        spots[p] = _encode_fjord(s, gs, p, fjord[k], need[p])
        other = 0
        for j in range(npl):
            if j != p:
                other = max(other, tl[j])
        for c in range(NF):
            fjord[k, c, FF_TURNS] = tl[p]
            fjord[k, c, FF_TURNS + 1] = other
    opp_tl = 0
    for k in range(1, npl):
        opp_tl = max(opp_tl, tl[seat[k]])

    comp = np.empty((MAXPL, NL), np.int64)
    size = np.zeros((MAXPL, NB), np.int64)
    adj = np.zeros((MAXPL, NB, 2), np.int64)
    for p in range(npl):
        _chains(s, gs, p, comp[p], size[p], adj[p])
    for k in range(5):
        land[NB + k, LF_INPLAY] = 1
        land[NB + k, LF_OCEAN] = 1
        if s[S_OCEAN + k] >= 0:
            land[NB + k, LF_SHIP] = 1
    gain = np.zeros(N_GAIN, np.int64)
    ogain = np.zeros(N_GAIN, np.int64)
    best = np.zeros((2, 2), np.int64)
    for i in range(NB):
        t = gs[G_TERR + i]
        if t == T_OUT:
            continue
        land[i, LF_INPLAY] = 1
        land[i, LF_TERRAIN + t] = 1
        occupied = False
        vo = 0
        for p in range(npl):
            v = s[S_VIK + p * NL + i]
            if v > 0:
                occupied = True
            if p != me:
                vo += v
        land[i, LF_VIK] = s[S_VIK + me * NL + i]
        land[i, LF_VIK + 1] = min(vo, 127)
        co = 0
        for k in range(1, npl):
            p = seat[k]
            if comp[p, i] >= 0:
                co = max(co, size[p, comp[p, i]])
        if comp[me, i] >= 0:
            land[i, LF_CHAIN] = min(size[me, comp[me, i]], 127)
        land[i, LF_CHAIN + 1] = min(co, 127)
        if t == T_TOWER:
            for k in range(npl):
                p = seat[k]
                links = s[S_PAIRS + p * MAXT + gs[G_TSLOT + i]]
                nl = 0
                while links:
                    nl += links & 1
                    links >>= 1
                touch = 0
                for d in range(6):
                    nb = LAND_NB[i, d]
                    if nb < NB and s[S_VIK + p * NL + nb] > 0:
                        touch = 1
                w = 0 if k == 0 else 1
                land[i, LF_TLINKS + w] = max(land[i, LF_TLINKS + w], nl)
                land[i, LF_TTOUCH + w] = max(land[i, LF_TTOUCH + w], touch)
        elif t == T_CASTLE:
            onext = 0
            for k in range(npl):
                p = seat[k]
                have = s[S_CASTLE + p * NL + i]
                nxt = 0
                if build and have < 3 and s[S_STOCK + i] > 0:
                    biggest = 0
                    for d in range(6):
                        nb = LAND_NB[i, d]
                        if nb < NB and comp[p, nb] >= 0:
                            biggest = max(biggest, size[p, comp[p, nb]])
                    nxt = max(1, 4 * (have + 1) - biggest)
                if k == 0:
                    land[i, LF_CLEVEL] = have
                    land[i, LF_CNEXT] = nxt
                else:
                    land[i, LF_CLEVEL + 1] = max(land[i, LF_CLEVEL + 1], have)
                    if nxt > 0 and (onext == 0 or nxt < onext):
                        onext = nxt
            land[i, LF_CNEXT + 1] = onext
        if t >= T_HOUSE:
            land[i, LF_STOCK + t - T_HOUSE] = s[S_STOCK + i]
        elif is_anchor(s, gs, i):
            if occupied:
                land[i, LF_STACK] = 1
            else:
                land[i, LF_FREE] = 1
                glob[GI_ANCHORS[t]] += 1
                for x in range(N_GAIN):
                    ogain[x] = 0
                obest0 = 0
                obest1 = 0
                for k in range(npl):
                    p = seat[k]
                    _placement_gain(s, gs, p, i, comp[p], size[p], adj[p], need[p], gain)
                    g0 = 1 + gain[1] + gain[2] + gain[3]
                    if k == 0:
                        for x in range(N_GAIN):
                            land[i, LF_GAIN + x] = min(gain[x], 127)
                        best[0, 0] = max(best[0, 0], g0)
                        best[0, 1] = max(best[0, 1], gain[4])
                    else:
                        for x in range(N_GAIN):
                            ogain[x] = max(ogain[x], gain[x])
                        obest0 = max(obest0, g0)
                        obest1 = max(obest1, gain[4])
                for x in range(N_GAIN):
                    land[i, LF_GAIN + N_GAIN + x] = min(ogain[x], 127)
                best[1, 0] = max(best[1, 0], obest0)
                best[1, 1] = max(best[1, 1], obest1)
    for i in range(NL):
        land[i, LF_TURNS] = tl[me]
        land[i, LF_TURNS + 1] = opp_tl

    # global: me and the strongest opponent
    top = seat[1]
    top_s = total_score(s, top)
    for k in range(2, npl):
        sc = total_score(s, seat[k])
        if sc > top_s:
            top = seat[k]
            top_s = sc
    _glob_player(s, me, glob, 0, spots[me], need[me], tl[me])
    _glob_player(s, top, glob, 1, spots[top], need[top], tl[top])
    for w in range(2):
        glob[GI_BEST_GAIN[w]] = best[w, 0]
        glob[GI_BEST_USEFUL[w]] = best[w, 1]
    glob[GI_TOMOVE] = 1 if (s[S_CUR] == me and s[S_OVER] == 0) else 0
    for i in range(5):
        o = s[S_TOWNER + i]
        if o == me:
            glob[GI_TROPHY[i, 0]] = 1
        elif o >= 0:
            glob[GI_TROPHY[i, 1]] = 1
    glob[GI_BAG] = s[S_BAGN]
    for k in range(5):
        sid = s[S_OCEAN + k]
        if sid >= 0:
            for r in range(4):
                glob[GI_OCEAN_RES[k, r]] += LS_NEED[sid, r]
            glob[GI_OCEAN_BONUS[k, LS_CAT[sid]]] = LS_BONUS[sid]
    glob[GI_PLAYERS[npl - 2]] = 1
    for k in range(1, npl):
        p = seat[k]
        glob[GI_SEAT[k - 1, 0]] = 1
        glob[GI_SEAT[k - 1, 1]] = total_score(s, p)
        glob[GI_SEAT[k - 1, 2]] = s[S_VLEFT + p]
        glob[GI_SEAT[k - 1, 3]] = tl[p]
        glob[GI_SEAT[k - 1, 4]] = 1 if p == top else 0


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
    fjord = np.zeros((MAXPL, NF, FJORD_F), np.int8)
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


# The network chooses the fjord layout too (see nn_bot.NNBot.TILE_K)
TILE_K, MAX_LAYOUTS, SHIP_K = 3, 4, 2
# With the network's proposals (its waar-kaart): the POL_K best spaces by the
# proposal, plus the GREEDY_HELP best by the greedy rule (a safety net while
# the proposals are learning; 0 = no greedy at all)
POL_K, POL_SHIP_K, GREEDY_HELP = 2, 2, 1


@njit(cache=True)
def _top(vs, n, k, order):
    """The k largest of vs[:n] (ties: the earlier one first) into order;
    returns how many."""
    used = np.zeros(n, np.bool_)
    m = min(k, n)
    for r in range(m):
        best = -1
        for j in range(n):
            if not used[j] and (best < 0 or vs[j] > vs[best]):
                best = j
        used[best] = True
        order[r] = best
    return m


@njit(cache=True)
def rank_tile_cells(s, gs, p, t, bw, noise, k, cells, scores):
    """The k best empty fjord spaces for a tile of type t by the greedy rule
    (best_tile_cell), best first. Returns how many."""
    cnt = np.zeros(7, np.int64)
    vs = np.empty(NF, np.float64)
    cs = np.empty(NF, np.int64)
    n_ = 0
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
        vs[n_] = v
        cs[n_] = c
        n_ += 1
    order = np.empty(k, np.int64)
    m = _top(vs, n_, k, order)
    for r in range(m):
        cells[r] = cs[order[r]]
        scores[r] = vs[order[r]]
    return m


@njit(cache=True)
def rank_ship_cells(s, p, sid, bw, noise, k, cells):
    """The k best empty fjord spaces for longship sid (best_ship_cell)."""
    cnt = np.zeros(7, np.int64)
    vs = np.empty(NF, np.float64)
    cs = np.empty(NF, np.int64)
    n_ = 0
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
        vs[n_] = v
        cs[n_] = c
        n_ += 1
    order = np.empty(k, np.int64)
    mm = _top(vs, n_, k, order)
    for r in range(mm):
        cells[r] = cs[order[r]]
    return mm


@njit(cache=True)
def _merge_cells(s, p, logits, col, kp, gcells, mg, cells, scores):
    """The kp best empty spaces of p's fjord by logits[:, col], then the
    greedy ones (gcells[:mg]) that aren't among them. scores = the logits."""
    vs = np.empty(NF, np.float64)
    cs = np.empty(NF, np.int64)
    n_ = 0
    for c in range(NF):
        if s[S_FJ + p * NF + c] == 0:
            vs[n_] = logits[c, col]
            cs[n_] = c
            n_ += 1
    order = np.empty(max(kp, 1), np.int64)
    m = _top(vs, n_, kp, order)
    for r in range(m):
        cells[r] = cs[order[r]]
        scores[r] = vs[order[r]]
    for r in range(mg):
        c = gcells[r]
        new = True
        for q in range(m):
            if cells[q] == c:
                new = False
        if new:
            cells[m] = c
            scores[m] = logits[c, col]
            m += 1
    return m


@njit(cache=True)
def policy_tile_cells(s, gs, p, t, bw, noise, tmap, cells, scores):
    """Spaces to try for a tile of type t: the proposal's best, plus greedy's."""
    gcells = np.empty(max(GREEDY_HELP, 1), np.int64)
    gsc = np.empty(max(GREEDY_HELP, 1), np.float64)
    mg = rank_tile_cells(s, gs, p, t, bw, noise, GREEDY_HELP, gcells, gsc)
    return _merge_cells(s, p, tmap, t, POL_K, gcells, mg, cells, scores)


@njit(cache=True)
def policy_ship_cells(s, p, sid, bw, noise, tmap, cells):
    """Spaces to try for longship sid: the proposal's best, plus greedy's."""
    gcells = np.empty(max(GREEDY_HELP, 1), np.int64)
    mg = rank_ship_cells(s, p, sid, bw, noise, GREEDY_HELP, gcells)
    scores = np.empty(POL_SHIP_K + GREEDY_HELP, np.float64)
    return _merge_cells(s, p, tmap, 7, POL_SHIP_K, gcells, mg, cells, scores)


@njit(cache=True)
def place_layouts(s, gs, cell, occ, dbl, bw, noise, out, k0, tmap, use_pol):
    """A Viking on `cell` with its tiles put on the fjord in up to
    MAX_LAYOUTS ways (nn_bot.NNBot._layouts): every tile tries its TILE_K
    best spaces; of all combinations, the best by the greedy rule's scores
    are kept. Written to out[k0:]; returns the new count."""
    W = MAX_LAYOUTS * max(TILE_K, POL_K + GREEDY_HELP)
    cur = np.empty((W, S_LEN), np.int32)
    ctype = np.empty((W, MAXP), np.int64)
    cfrom = np.empty((W, MAXP), np.int64)
    cn = np.zeros(W, np.int64)
    csc = np.zeros(W, np.float64)
    nxt = np.empty((W, S_LEN), np.int32)
    ntype = np.empty((W, MAXP), np.int64)
    nfrom = np.empty((W, MAXP), np.int64)
    nn = np.zeros(W, np.int64)
    nsc = np.zeros(W, np.float64)
    cells = np.empty(max(TILE_K, POL_K + GREEDY_HELP), np.int64)
    scores = np.empty(max(TILE_K, POL_K + GREEDY_HELP), np.float64)
    order = np.empty(MAX_LAYOUTS, np.int64)
    cur[0] = s
    cn[0] = place_viking(cur[0], gs, cell, occ, dbl, ctype[0], cfrom[0])
    nl = 1
    while cn[0] > 0:
        m = 0
        for l in range(nl):
            p = cur[l, S_CUR]
            n = cn[l]
            j = 0
            if n > fj_empty_count(cur[l], p):          # not everything fits: best first
                best = -1.0
                for q in range(n):
                    if KEEP[ctype[l, q]] > best:
                        best = KEEP[ctype[l, q]]
                        j = q
            if use_pol:
                mk = policy_tile_cells(cur[l], gs, p, ctype[l, j], bw, noise, tmap, cells, scores)
            else:
                mk = rank_tile_cells(cur[l], gs, p, ctype[l, j], bw, noise, TILE_K, cells, scores)
            for r in range(mk):
                nxt[m] = cur[l]
                ntype[m] = ctype[l]
                nfrom[m] = cfrom[l]
                nn[m] = place_tile(nxt[m], gs, ntype[m], nfrom[m], n, j, cells[r])
                nsc[m] = csc[l] + scores[r]
                m += 1
        nl = _top(nsc, m, MAX_LAYOUTS, order)
        for r in range(nl):
            q = order[r]
            cur[r] = nxt[q]
            ctype[r] = ntype[q]
            cfrom[r] = nfrom[q]
            cn[r] = nn[q]
            csc[r] = nsc[q]
    for l in range(nl):
        out[k0 + l] = cur[l]
    return k0 + nl


@njit(cache=True)
def firsts(s, gs, bw, noise, placed, plain, tmap, use_pol, row_opt, opt_cell, opt_var):
    """Every Viking placement of the player to move: `placed` gets the state
    after placing (and putting the tiles away), `plain` the same state with
    the turn simply ended (for a quick first judgement). Returns how many."""
    cells = np.empty(MAXO, np.int64)
    occs = np.empty(MAXO, np.int64)
    dbls = np.empty(MAXO, np.int64)
    n = placement_options(s, gs, cells, occs, dbls)
    k = 0
    for o in range(n):
        k0 = k
        k = place_layouts(s, gs, cells[o], occs[o] == 1, dbls[o] == 1, bw, noise, placed, k,
                          tmap, use_pol)
        for q in range(k0, k):
            row_opt[q] = o                         # which placement each row is
        opt_cell[o] = cells[o]
        opt_var[o] = 2 if occs[o] == 1 else (1 if dbls[o] == 1 else 0)
    for q in range(k):
        plain[q] = placed[q]
        end_turn(plain[q], gs)
    return k, n


@njit(cache=True)
def endings(st, gs, bw, noise, shuffle, out, k, tmap, use_pol):
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
    cells = np.empty(max(SHIP_K, POL_SHIP_K + GREEDY_HELP), np.int64)
    for j in range(ns):
        slot = slots[j]
        g3 = st.copy()
        nw = 1
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
            if use_pol:
                nw = policy_ship_cells(g3, p, g3[S_OCEAN + slot], bw, noise, tmap, cells)
            else:
                nw = rank_ship_cells(g3, p, g3[S_OCEAN + slot], bw, noise, SHIP_K, cells)
        for w in range(nw):
            g4 = g3.copy()
            if slot >= 0:
                take_ship(g4, gs, slot, cells[w])
            tr = best_claimable(g4, gs)
            for option in range(2 if tr >= 0 else 1):
                out[k] = g4
                if option == 1:
                    claim(out[k], tr)
                end_turn(out[k], gs)
                k += 1
    return k


# ---------------------------------------------------------------------------
# What the proposals ("policy") learn: where the player's search ended up.
# Every candidate turn gets a weight from its value, softmax(value / TAU);
# the moves of each candidate (read from the difference between the state at
# the start of the turn and the candidate's end state) get that weight.
# ---------------------------------------------------------------------------

POLICY_TAU = 1.0            # points: candidates within ~1 point share the weight


@njit(cache=True)
def policy_target(s0, s1, w, place, tile, ship):
    """Add weight w to the moves that lead from s0 (start of the turn) to s1
    (a candidate's end): place[NL, 3] Viking spaces (plain, double shield,
    stacked), tile[NF, 8] fjord spaces per item (7) and for a new longship,
    ship[6] the ocean slot taken (5 = none)."""
    p = s0[S_CUR]
    dbl = s0[S_SHIELD + p * 3 + 2] > 0 and s1[S_SHIELD + p * 3 + 2] == 0
    first = True
    for i in range(NB):
        if s1[S_VIK + p * NL + i] > s0[S_VIK + p * NL + i]:
            if s0[S_VIK + i] + s0[S_VIK + NL + i] > 0:
                v = 2                                   # stacked (occupy shield)
            elif dbl and first:
                v = 1                                   # double shield
            else:
                v = 0
            place[i, v] += w
            first = False
    took = False
    for c in range(NF):
        c0 = s0[S_FJ + p * NF + c]
        c1 = s1[S_FJ + p * NF + c]
        if c0 == 0 and c1 != 0:
            if c1 <= 7:
                tile[c, c1 - 1] += w
            elif c1 == FJ_SHIP:
                tile[c, 7] += w
                sid = s1[S_FJSHIP + p * NF + c]
                for k in range(5):
                    if s0[S_OCEAN + k] == sid:
                        ship[k] += w
                        took = True
    if not took:
        ship[5] += w


def search_target(s, out, k, vals):
    """The proposals' target for one turn: the weighted moves of all k
    candidates the search judged. Returns (place, tile, ship)."""
    wts = np.exp((vals[:k] - vals[:k].max()) / POLICY_TAU)
    wts /= wts.sum()
    place = np.zeros((NL, 3))
    tile = np.zeros((NF, 8))
    ship = np.zeros(6)
    for c in range(k):
        policy_target(s, out[c], wts[c], place, tile, ship)
    return place, tile, ship


def proposal_maps(net, requests):
    """The network's proposals for positions at the start of a turn.
    requests: list of (state, gs, me). Returns per request (place [NL, 3],
    tile [NF, 8]) as float64, or None for a network without proposals."""
    if not getattr(net, "has_proposals", False):
        return [None] * len(requests)
    n = len(requests)
    land = np.zeros((n, NL, LAND_F), np.int8)
    fjord = np.zeros((n, MAXPL, NF, FJORD_F), np.int8)
    glob = np.zeros((n, GLOB_F), np.int16)
    for i, (st, gs, me) in enumerate(requests):
        encode(st, gs, me, land[i], fjord[i], glob[i])
    maps = net.proposals(land, fjord, glob)
    return [(np.asarray(maps["pol_place"][i], np.float64), np.asarray(maps["pol_tile"][i], np.float64))
            for i in range(n)]


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
            v[k] = outcome(states[k], gs, me)[0]
        todo = np.where(~over)[0]
        rows.append((states, gs, me, todo))
        vals.append(v)
    total = sum(len(r[3]) for r in rows)
    if total:
        land = np.zeros((total, NL, LAND_F), np.int8)
        fjord = np.zeros((total, MAXPL, NF, FJORD_F), np.int8)
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
    TOP_POL = 2             # + the best placements by the proposal (plaats-kaart)

    def __init__(self, net, rng, explore=0.0, noise=True, shuffle=True, weights=None):
        self.net = net
        self.rng = rng
        self.explore = explore
        self.noise = noise
        self.shuffle = shuffle
        self.bw = bot_weights(weights)
        # placed / plain states of the Viking placements, the same for a
        # second Viking (shield), and the finished-turn candidates
        n = MAXO * MAX_LAYOUTS
        self.buf = [np.empty((m, S_LEN), np.int32) for m in (n, n, n, n, 4096)]
        self.row_opt = np.zeros(n, np.int64)
        self.opt_cell = np.zeros(MAXO, np.int64)
        self.opt_var = np.zeros(MAXO, np.int64)
        self.use_policy = getattr(net, "has_proposals", False)
        self._no_map = np.zeros((NF, 8))

    def turn_steps(self, s, gs):
        """One turn as a generator. Whenever it needs the network, it yields
        (states, n, me) and expects the values of states[0:n] back through
        send(). It ends by returning the state after the turn.

        Written this way, one implementation serves both: turn() below
        answers each question at once, and selfplay_batched() collects the
        questions of many games to ask the network (on the GPU) together."""
        me = int(s[S_CUR])
        self.target = None
        placed, plain, placed2, plain2, out = self.buf
        # the proposals first (a request with no states: "your proposals for s")
        pmap = None
        if self.use_policy:
            pmap = yield (None, s, me)
        use_pol = pmap is not None
        tmap = pmap[1] if use_pol else self._no_map
        if s[S_PHASE] == PH_PLACE:
            n, n_opt = firsts(s, gs, self.bw, self.noise, placed, plain, tmap, use_pol,
                              self.row_opt, self.opt_cell, self.opt_var)
            if n <= self.TOP_FULL:
                idx = list(range(n))
            else:
                vals = yield (plain, n, me)
                idx = list(np.argsort(-vals)[:self.TOP_FULL])
                if use_pol:
                    # also the placements the proposal likes best (with their
                    # best layout), even if the quick look without a longship
                    # didn't rank them high
                    score = pmap[0][self.opt_cell[:n_opt], self.opt_var[:n_opt]]
                    for o in np.argsort(-score, kind="stable")[:self.TOP_POL]:
                        rows = np.where(self.row_opt[:n] == o)[0]
                        r = int(rows[np.argmax(vals[rows])])
                        if r not in idx:
                            idx.append(r)
            chosen = [placed[i].copy() for i in idx]
        else:
            chosen = [s.copy()]
        k = 0
        for st in chosen:
            k = endings(st, gs, self.bw, self.noise, self.shuffle, out, k, tmap, use_pol)
            if s[S_PHASE] == PH_PLACE and can_use_extra(st, gs):
                sx = st.copy()
                use_extra(sx)
                n2, _ = firsts(sx, gs, self.bw, self.noise, placed2, plain2, tmap, use_pol,
                               self.row_opt, self.opt_cell, self.opt_var)
                if n2 <= self.TOP_EXTRA:
                    idx2 = np.arange(n2)
                else:
                    vals = yield (plain2, n2, me)
                    idx2 = np.argsort(-vals)[:self.TOP_EXTRA]
                for i in idx2:
                    k = endings(placed2[i].copy(), gs, self.bw, self.noise, self.shuffle, out, k,
                                tmap, use_pol)
        if k == 0:
            s2 = s.copy()
            end_turn(s2, gs)
            return s2
        vals = yield (out, k, me)
        self.target = search_target(s, out, k, np.asarray(vals))
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
                if states is None:                     # proposals for position n
                    req = steps.send(proposal_maps(self.net, [(n, gs, me)])[0])
                else:
                    req = steps.send(judge_states(self.net, [(states, n, gs, me)])[0])
        except StopIteration as done:
            return done.value


def record(states, gs, seed_, targets=None):
    """Training data of a finished game: every turn-start position seen by
    every player, labelled with how the game ended for that player. Rows:
    turn 0 (player 0, 1, ...), turn 1 (player 0, 1, ...), ... - the format
    of gen_data.play_and_record. (The proposals' targets are for 2
    players only so far: `targets` is not used here.)"""
    n = len(states)
    npl = int(gs[G_NP])
    arr = np.stack(states)
    land = np.zeros((npl * n, NL, LAND_F), np.int8)
    fjord = np.zeros((npl * n, MAXPL, NF, FJORD_F), np.int8)
    glob = np.zeros((npl * n, GLOB_F), np.int16)
    for me in range(npl):
        l, f, gl = land[me::npl], fjord[me::npl], glob[me::npl]
        encode_many(arr, n, gs, me, l, f, gl)
    res = [outcome(states[-1], gs, me) for me in range(npl)]
    margin = np.array([res[k % npl][0] for k in range(npl * n)], np.float32)
    win = np.array([res[k % npl][1] for k in range(npl * n)], np.float32)
    game = np.full(npl * n, seed_, np.int64)
    return land, fjord, glob, margin, win, game


def selfplay_game(seed_, layout, rules, net, explore=0.0, noise=True, players=4):
    """One game between two network players, played on its own. Returns
    (land, fjord, glob, margin, win, game) like gen_data.play_and_record."""
    import random
    g = L.Game([("p%d" % i, True) for i in range(players)], seed_, layout, L.rules_for(rules))
    s, gs = from_game(g)
    seed(seed_ % (2 ** 31))
    player = FastPlayer(net, random.Random(seed_ * 7), explore, noise=noise, shuffle=noise)
    states = [s]
    targets = []
    while s[S_OVER] == 0:
        s = player.turn(s, gs)
        states.append(s)
        targets.append(player.target)
        assert len(states) < 400
    return record(states, gs, seed_, targets)


def selfplay_batched(seeds, layout, rules, net, explore=0.0, parallel=64, noise=True, players=4):
    """Self-play of many games at once, in step: every game runs until its
    player needs the network, then the positions of ALL games are judged in
    one batch (on the GPU with gpu_net.TorchNet), and every game gets its answers.
    Yields the training data of each game as it finishes (see record())."""
    import random
    todo = list(seeds)
    seed((todo[0] if todo else 0) % (2 ** 31))
    slots = []                  # one per game being played

    def start(sd, player=None):
        g = L.Game([("p%d" % i, True) for i in range(players)], sd, layout, L.rules_for(rules))
        s, gs = from_game(g)
        rng = random.Random(sd * 7)
        if player is None:
            player = FastPlayer(net, rng, explore, noise=noise, shuffle=noise)
        player.rng = rng
        return {"seed": sd, "s": s, "gs": gs, "player": player, "states": [s],
                "targets": [], "steps": None, "req": None}

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
                slot["targets"].append(slot["player"].target)
                slot["steps"] = None
                slot["req"] = None
                answer = None
                if done.value[S_OVER]:
                    return record(slot["states"], slot["gs"], slot["seed"], slot["targets"])

    while todo or slots:
        while todo and len(slots) < parallel:
            slot = start(todo.pop(0))
            data = advance(slot, None)
            if data is not None:               # (a game can't end without a question)
                yield data
                continue
            slots.append(slot)
        # two kinds of questions: proposals (req[0] is None) and judgements
        pol = [sl for sl in slots if sl["req"][0] is None]
        val = [sl for sl in slots if sl["req"][0] is not None]
        got = {}
        for sl, ans in zip(pol, proposal_maps(net, [(sl["req"][1], sl["gs"], sl["req"][2]) for sl in pol])):
            got[id(sl)] = ans
        for sl, ans in zip(val, judge_states(net, [(sl["req"][0], sl["req"][1], sl["gs"], sl["req"][2])
                                                   for sl in val])):
            got[id(sl)] = ans
        answers = [got[id(sl)] for sl in slots]
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
