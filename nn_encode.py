"""
Turn a Looot position into numbers for the neural network (the "encoding").

A position is always seen from one player, "me". The encoding has three
parts, all whole numbers so they can be stored compactly:

  land   [N_LAND, LAND_F]      one row per landscape/ocean space
  fjord  [2, N_FJORD, FJORD_F] my fjord, then the opponent's fjord
  glob   [GLOB_F]              everything that isn't a space on a board

For now this is written for 2-player games (N_PLAYERS = 2): the gameboard is
boards 1 and 2 plus the ocean, always in the same places. Only which side of
each board comes up changes, and that is in the terrain features.
"""

import numpy as np

import looot as L

N_PLAYERS = 2

# ---------------------------------------------------------------------------
# Fixed geometry: which space is row i, and who its neighbours are
# ---------------------------------------------------------------------------

LAND_CELLS = [c for b in L.board_cells(N_PLAYERS) for c in b] + list(L.OCEAN_CELLS)
LAND_INDEX = {c: i for i, c in enumerate(LAND_CELLS)}
N_LAND = len(LAND_CELLS)

FJORD_CELLS, _ = L.fjord_cells()
FJORD_INDEX = {c: i for i, c in enumerate(FJORD_CELLS)}
N_FJORD = len(FJORD_CELLS)


def neighbour_table(cells, index):
    """[len(cells), 6] array: the row of each of the 6 neighbours, or
    len(cells) (a padding row of zeros) where there is no neighbour.
    Direction d is always L.DIRS[d], so the network can tell them apart."""
    pad = len(cells)
    t = np.full((len(cells), 6), pad, dtype=np.int64)
    for i, (q, r) in enumerate(cells):
        for d, (dq, dr) in enumerate(L.DIRS):
            t[i, d] = index.get((q + dq, r + dr), pad)
    return t


LAND_NB = neighbour_table(LAND_CELLS, LAND_INDEX)
FJORD_NB = neighbour_table(FJORD_CELLS, FJORD_INDEX)

# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------

TERRAINS = ["forest", "field", "mountain", "battlefield", "house", "watchtower", "castle"]
RES = ["wood", "sheep", "gold", "axe"]
BLD = ["house", "watchtower", "castle"]
CATS = ["castle", "watchtower", "house", "gold", "sheep", "wood"]
SITE_ITEMS = RES + BLD

# land features
LF_TERRAIN = 0                  # 7 one-hot
LF_OCEAN = 7
LF_SHIP = 8                     # ocean space with a longship on it
LF_MY_VIK, LF_OPP_VIK = 9, 10   # number of Vikings
LF_STOCK = 11                   # building tiles left on this space
LF_MY_CHAIN, LF_OPP_CHAIN = 12, 13   # size of the chain this space is in
LF_MY_CASTLE, LF_OPP_CASTLE = 14, 15  # tiles already taken from this castle
LF_MY_TOWERS, LF_OPP_TOWERS = 16, 17  # towers this tower is linked to
LF_ANCHOR = 18                  # free resource space next to a Viking/longship
LAND_F = 19

# fjord features
FF_EMPTY = 0
FF_RES = 1                      # 4 one-hot
FF_BLD = 5                      # 3 one-hot
FF_SHIP, FF_SHIP_DONE = 8, 9
FF_SHIP_MISSING = 10            # 4: resources still missing (count)
FF_SHIP_BONUS = 14              # 6: bonus value in the slot of its category
FF_SITE, FF_SITE_DONE = 20, 21
FF_SITE_MISSING = 22            # 7: items still missing (count)
FF_SITE_VP = 29
# helper features (version 2): counting the network would otherwise have to learn
FF_ROOM = 30                    # empty neighbouring spaces (every space)
FF_FILLABLE = 31                # unfinished ship/site that can still be finished
FF_SLACK = 32                   # unfinished ship/site: Vikings left - items missing
FJORD_F = 33

ENCODING_VERSION = 2

# global features
GLOB_NAMES = (["to_move_me", "vik_me", "vik_opp"] +
              ["shield_me_%s" % s for s in ("extra", "occupy", "double")] +
              ["shield_opp_%s" % s for s in ("extra", "occupy", "double")] +
              ["trophy%d_me" % i for i in range(5)] +
              ["trophy%d_opp" % i for i in range(5)] +
              ["axes_me", "axes_opp"] +
              ["%s_%s_%s" % (w, c, p) for p in ("me", "opp") for c in CATS
               for w in ("count", "value")] +
              ["sites_me", "sites_opp", "unfilled_me", "unfilled_opp",
               "total_me", "total_opp", "room_me", "room_opp", "bag"] +
              ["ocean%d_%s" % (k, f) for k in range(5)
               for f in RES + ["bonus_" + c for c in CATS]] +
              # version 2: free spaces to place a Viking, per resource type,
              # and fjord spaces where a new longship could still be filled
              ["anchors_%s" % r for r in RES] +
              ["ship_spots_me", "ship_spots_opp"])
GLOB_F = len(GLOB_NAMES)
G = {n: i for i, n in enumerate(GLOB_NAMES)}


def _missing(need, have):
    miss = list(need)
    for h in have:
        if h in miss:
            miss.remove(h)
    return miss


def _chains(g, idx):
    """{cell: size of player idx's chain through that cell}"""
    size = {}
    for c, owners in g.vikings.items():
        if idx in owners and c not in size:
            ch = g.chain(g.players[idx], c)
            for cc in ch:
                size[cc] = len(ch)
    return size


def _tower_links(p):
    links = {}
    for pk in p.wt_pairs:
        a, b = pk.split("|")
        links[a] = links.get(a, 0) + 1
    return links


def _empty_neighbours(p, c):
    return sum(1 for n in L.neighbours(c) if n in FJORD_INDEX and n not in p.fjord)


def _fjord_reference(p, out):
    """Fill out [N_FJORD, FJORD_F] for player p. Returns the number of
    empty spaces where a new longship could still be filled."""
    left = p.vikings_left
    # resources a player can still gain: 1 per Viking, +1 with the double shield
    gain = left + (1 if p.shields.get("double") else 0)
    ship_spots = 0
    for c, i in FJORD_INDEX.items():
        it = p.fjord.get(c)
        room = _empty_neighbours(p, c)
        out[i, FF_ROOM] = room
        if it is None:
            out[i, FF_EMPTY] = 1
            # a new longship here needs 3 resources: next to it already, or room
            have = sum(1 for h in p.adjacent_items(c) if h in RES)
            if min(have, 3) + room >= 3 and 3 - min(have, 3) <= gain:
                ship_spots += 1
            continue
        k = it["kind"]
        if k == "res":
            out[i, FF_RES + RES.index(it["type"])] = 1
        elif k == "bld":
            out[i, FF_BLD + BLD.index(it["type"])] = 1
        elif k == "ship":
            out[i, FF_SHIP_DONE if it["filled"] else FF_SHIP] = 1
            out[i, FF_SHIP_BONUS + CATS.index(it["cat"])] = it["bonus"]
            if not it["filled"]:
                miss = _missing(it["need"], p.adjacent_items(c))
                for m in miss:
                    out[i, FF_SHIP_MISSING + RES.index(m)] += 1
                out[i, FF_FILLABLE] = 1 if len(miss) <= room and len(miss) <= gain else 0
                out[i, FF_SLACK] = max(-13, min(13, gain - len(miss)))
        elif k == "site":
            out[i, FF_SITE_DONE if it["done"] else FF_SITE] = 1
            out[i, FF_SITE_VP] = L.SITE_VP[it["type"]]
            if not it["done"]:
                miss = _missing(it["need"], p.adjacent_items(c))
                for m in miss:
                    out[i, FF_SITE_MISSING + SITE_ITEMS.index(m)] += 1
                # buildings come without a Viking of their own, so only room counts
                n_res = sum(1 for m in miss if m in RES)
                out[i, FF_FILLABLE] = 1 if len(miss) <= room and n_res <= gain else 0
                out[i, FF_SLACK] = max(-13, min(13, gain - len(miss)))
    return ship_spots


def encode_reference(g, me):
    """The straightforward version of encode(), kept to test the fast one.
    Encode game g as seen by player index `me`.
    Returns (land int8, fjord int8, glob int16)."""
    assert len(g.players) == N_PLAYERS
    opp = 1 - me
    land = np.zeros((N_LAND, LAND_F), dtype=np.int8)
    fjord = np.zeros((2, N_FJORD, FJORD_F), dtype=np.int8)
    glob = np.zeros(GLOB_F, dtype=np.int16)

    # --- landscape and ocean ---
    chain_me, chain_opp = _chains(g, me), _chains(g, opp)
    pm, po = g.players[me], g.players[opp]
    links_me, links_opp = _tower_links(pm), _tower_links(po)
    for c, t in g.land.items():
        i = LAND_INDEX[c]
        land[i, LF_TERRAIN + TERRAINS.index(t)] = 1
        v = g.vikings.get(c, [])
        land[i, LF_MY_VIK] = v.count(me)
        land[i, LF_OPP_VIK] = v.count(opp)
        land[i, LF_STOCK] = g.stock.get(c, 0)
        land[i, LF_MY_CHAIN] = min(chain_me.get(c, 0), 127)
        land[i, LF_OPP_CHAIN] = min(chain_opp.get(c, 0), 127)
        k = L.key(c)
        land[i, LF_MY_CASTLE] = pm.castle_taken.get(k, 0)
        land[i, LF_OPP_CASTLE] = po.castle_taken.get(k, 0)
        land[i, LF_MY_TOWERS] = links_me.get(k, 0)
        land[i, LF_OPP_TOWERS] = links_opp.get(k, 0)
        if t in L.TERRAIN_RESOURCE and not v and g.is_anchor(c):
            land[i, LF_ANCHOR] = 1
            glob[G["anchors_" + L.TERRAIN_RESOURCE[t]]] += 1
    for slot, c in enumerate(g.ocean):
        i = LAND_INDEX[c]
        land[i, LF_OCEAN] = 1
        land[i, LF_SHIP] = 1 if g.ocean_ships[slot] is not None else 0

    # --- fjords ---
    glob[G["ship_spots_me"]] = _fjord_reference(pm, fjord[0])
    glob[G["ship_spots_opp"]] = _fjord_reference(po, fjord[1])

    # --- everything else ---
    glob[G["to_move_me"]] = 1 if (g.current == me and not g.game_over) else 0
    for who, p in (("me", pm), ("opp", po)):
        glob[G["vik_" + who]] = p.vikings_left
        for s in ("extra", "occupy", "double"):
            glob[G["shield_%s_%s" % (who, s)]] = 1 if p.shields[s] else 0
        glob[G["axes_" + who]] = g.axes(p)
        sc = p.score()
        for cat in CATS:
            glob[G["count_%s_%s" % (cat, who)]] = sc["rows"][cat]["count"]
            glob[G["value_%s_%s" % (cat, who)]] = sc["rows"][cat]["value"]
        glob[G["sites_" + who]] = sc["sites"]
        glob[G["unfilled_" + who]] = sc["unfilled"]
        glob[G["total_" + who]] = sc["total"]
        glob[G["room_" + who]] = len(p.empty_cells())
    for i, owner in enumerate(g.trophy_owner):
        if owner == me:
            glob[G["trophy%d_me" % i]] = 1
        elif owner == opp:
            glob[G["trophy%d_opp" % i]] = 1
    glob[G["bag"]] = len(g.bag)
    for k, sid in enumerate(g.ocean_ships):
        if sid is None:
            continue
        need, cat, bonus = L.LONGSHIPS[sid]
        for r in need:
            glob[G["ocean%d_%s" % (k, r)]] += 1
        glob[G["ocean%d_bonus_%s" % (k, cat)]] = bonus
    return land, fjord, glob


def outcome(g, me):
    """Training labels once the game is over: (margin, win) for `me`."""
    s = [p.score()["total"] for p in g.players]
    margin = s[me] - max(s[j] for j in range(len(s)) if j != me)
    w = g.winners()
    win = (1.0 / len(w)) if me in w else 0.0
    return float(margin), float(win)


# ---------------------------------------------------------------------------
# The fast encoder (the one the players use)
# ---------------------------------------------------------------------------

KEY_INDEX = {L.key(c): i for c, i in LAND_INDEX.items()}
FJ_IDX = FJORD_INDEX
N_ITEMS = len(SITE_ITEMS)                      # wood sheep gold axe house watchtower castle
ITEM_INDEX = {t: k for k, t in enumerate(SITE_ITEMS)}
RES_SET = set(RES)
CAT_INDEX = {c: k for k, c in enumerate(CATS)}
CAT_ITEM = [ITEM_INDEX[c] for c in CATS]       # item column of each score category
BASE = np.array([L.BASE_VALUE[c] for c in CATS], dtype=np.int64)
OCEAN_IDX = np.array([LAND_INDEX[c] for c in L.OCEAN_CELLS])

_static = {}          # id(land dict) -> (land dict, base array, is_res, res_type)


def _board_static(g):
    """Terrain features and resource masks, computed once per gameboard.
    Copies of a game share the same land dict, so the cache stays small."""
    hit = _static.get(id(g.land))
    if hit is not None and hit[0] is g.land:
        return hit
    base = np.zeros((N_LAND, LAND_F), dtype=np.int8)
    is_res = np.zeros(N_LAND + 1, dtype=bool)
    res_type = np.zeros(N_LAND, dtype=np.int64)
    for c, t in g.land.items():
        i = LAND_INDEX[c]
        base[i, LF_TERRAIN + TERRAINS.index(t)] = 1
        if t in L.TERRAIN_RESOURCE:
            is_res[i] = True
            res_type[i] = RES.index(L.TERRAIN_RESOURCE[t])
    base[OCEAN_IDX, LF_OCEAN] = 1
    if len(_static) > 64:
        _static.clear()
    hit = _static[id(g.land)] = (g.land, base, is_res, res_type)
    return hit


LAND_NB_LIST = [[int(n) for n in row if n < N_LAND] for row in LAND_NB]

# positions in the global vector, looked up once instead of on every call
GI = {who: {"spots": G["ship_spots_" + who], "vik": G["vik_" + who],
            "shields": [G["shield_%s_%s" % (who, s)] for s in ("extra", "occupy", "double")],
            "axes": G["axes_" + who],
            "count": [G["count_%s_%s" % (c, who)] for c in CATS],
            "value": [G["value_%s_%s" % (c, who)] for c in CATS],
            "sites": G["sites_" + who], "unfilled": G["unfilled_" + who],
            "total": G["total_" + who], "room": G["room_" + who]}
      for who in ("me", "opp")}
GI_ANCHORS = [G["anchors_" + r] for r in RES]
GI_TROPHY = [(G["trophy%d_me" % i], G["trophy%d_opp" % i]) for i in range(len(L.TROPHIES))]
GI_OCEAN = [({r: G["ocean%d_%s" % (k, r)] for r in RES},
             {c: G["ocean%d_bonus_%s" % (k, c)] for c in CATS}) for k in range(5)]


def _chain_sizes(cells):
    """cells: set of land indices with this player's Vikings.
    Returns {index: size of its chain}."""
    size = {}
    for s in cells:
        if s in size:
            continue
        seen, todo = {s}, [s]
        while todo:
            c = todo.pop()
            for n in LAND_NB_LIST[c]:
                if n in cells and n not in seen:
                    seen.add(n)
                    todo.append(n)
        for c in seen:
            size[c] = len(seen)
    return size


FJ_NB_LIST = [[int(n) for n in row if n < N_FJORD] for row in FJORD_NB]
BASE_LIST = [L.BASE_VALUE[c] for c in CATS]


def _fjord_fast(p, out):
    """Same as nn_encode._fjord, plus the score numbers of p.
    Small arrays: plain Python lists, written into `out` in a few bulk steps.
    Returns (ship_spots, score dict)."""
    item = [-1] * N_FJORD                # item type index, or -1
    empty = [True] * N_FJORD
    special = []
    bonus = [0] * len(CATS)
    counts = [0] * N_ITEMS
    sites_vp = unfilled = 0
    for c, it in p.fjord.items():
        i = FJ_IDX[c]
        empty[i] = False
        k = it["kind"]
        if k == "res" or k == "bld":
            t = ITEM_INDEX[it["type"]]
            item[i] = t
            counts[t] += 1
        else:
            special.append((i, it))
            if k == "ship":
                if it["filled"]:
                    bonus[CAT_INDEX[it["cat"]]] += it["bonus"]
                else:
                    unfilled += 1
            elif it["done"]:
                sites_vp += L.SITE_VP[it["type"]]

    gain = p.vikings_left + (1 if p.shields.get("double") else 0)
    room = [0] * N_FJORD
    spots = 0
    for i in range(N_FJORD):
        nb = FJ_NB_LIST[i]
        r = 0
        for n in nb:
            if empty[n]:
                r += 1
        room[i] = r
        if empty[i]:
            # a new longship here needs 3 resources: next to it already, or room
            h = 0
            for n in nb:
                if 0 <= item[n] < 4:
                    h += 1
            h = min(h, 3)
            if h + r >= 3 and 3 - h <= gain:
                spots += 1

    # bulk writes: room, empty, one-hot of items
    out[:, FF_ROOM] = room
    out[:, FF_EMPTY] = empty
    rows = [i for i in range(N_FJORD) if item[i] >= 0]
    if rows:
        cols = [(FF_RES + item[i]) if item[i] < 4 else (FF_BLD + item[i] - 4) for i in rows]
        out[rows, cols] = 1

    for i, it in special:
        adj = [0] * N_ITEMS
        for n in FJ_NB_LIST[i]:
            if item[n] >= 0:
                adj[item[n]] += 1
        miss = [0] * N_ITEMS
        for t in it["need"]:
            k = ITEM_INDEX[t]
            if adj[k]:
                adj[k] -= 1
            else:
                miss[k] += 1
        nmiss = sum(miss)
        row = out[i]
        if it["kind"] == "ship":
            row[FF_SHIP_DONE if it["filled"] else FF_SHIP] = 1
            row[FF_SHIP_BONUS + CAT_INDEX[it["cat"]]] = it["bonus"]
            if not it["filled"]:
                row[FF_SHIP_MISSING:FF_SHIP_MISSING + 4] = miss[:4]
                row[FF_FILLABLE] = 1 if nmiss <= room[i] and nmiss <= gain else 0
                row[FF_SLACK] = max(-13, min(13, gain - nmiss))
        else:
            row[FF_SITE_DONE if it["done"] else FF_SITE] = 1
            row[FF_SITE_VP] = L.SITE_VP[it["type"]]
            if not it["done"]:
                row[FF_SITE_MISSING:FF_SITE_MISSING + N_ITEMS] = miss
                n_res = miss[0] + miss[1] + miss[2] + miss[3]
                row[FF_FILLABLE] = 1 if nmiss <= room[i] and n_res <= gain else 0
                row[FF_SLACK] = max(-13, min(13, gain - nmiss))

    cat_count = [counts[k] for k in CAT_ITEM]
    value = [BASE_LIST[k] + bonus[k] for k in range(len(CATS))]
    trophy = L.TROPHIES[p.trophy][1] if p.trophy is not None else 0
    total = sum(v * n for v, n in zip(value, cat_count)) + sites_vp + trophy - 5 * unfilled
    score = {"count": cat_count, "value": value, "sites": sites_vp,
             "unfilled": unfilled, "total": total, "axes": counts[3],
             "room": N_FJORD - len(p.fjord)}
    return spots, score


def encode(g, me):
    """Encode game g as seen by player index `me`: (land int8, fjord int8,
    glob int16). Exactly the numbers of encode_reference(), about 3x faster:
    what never changes during a game is worked out once per board, and
    each fjord is counted in one pass."""
    opp = 1 - me
    _, base, is_res, res_type = _board_static(g)
    land = base.copy()
    fjord = np.zeros((2, N_FJORD, FJORD_F), dtype=np.int8)
    glob = np.zeros(GLOB_F, dtype=np.int16)

    # --- Vikings, chains ---
    occ = np.zeros(N_LAND + 1, dtype=bool)
    mine, theirs = set(), set()
    for c, v in g.vikings.items():
        i = LAND_INDEX[c]
        a, b = v.count(me), v.count(opp)
        land[i, LF_MY_VIK] = a
        land[i, LF_OPP_VIK] = b
        if v:
            occ[i] = True
        if a:
            mine.add(i)
        if b:
            theirs.add(i)
    for s, col in ((mine, LF_MY_CHAIN), (theirs, LF_OPP_CHAIN)):
        for i, n in _chain_sizes(s).items():
            land[i, col] = min(n, 127)

    # --- building stacks, castles, towers ---
    for c, n in g.stock.items():
        land[LAND_INDEX[c], LF_STOCK] = n
    pm, po = g.players[me], g.players[opp]
    for p, col in ((pm, LF_MY_CASTLE), (po, LF_OPP_CASTLE)):
        for k, n in p.castle_taken.items():
            land[KEY_INDEX[k], col] = n
    for p, col in ((pm, LF_MY_TOWERS), (po, LF_OPP_TOWERS)):
        for pk in p.wt_pairs:
            land[KEY_INDEX[pk.split("|")[0]], col] += 1   # as in nn_encode._tower_links

    # --- where a Viking may be placed ---
    source = occ.copy()
    no_ships = not g.rules["longships"]
    for slot, sid in enumerate(g.ocean_ships):
        i = OCEAN_IDX[slot]
        if sid is not None:
            land[i, LF_SHIP] = 1
        if sid is not None or no_ships:
            source[i] = True
    anchor = is_res[:N_LAND] & ~occ[:N_LAND] & source[LAND_NB].any(1)
    land[:, LF_ANCHOR] = anchor
    glob[GI_ANCHORS] = np.bincount(res_type[anchor], minlength=4)

    # --- fjords and scores ---
    for slot_p, p, who in ((0, pm, "me"), (1, po, "opp")):
        spots, sc = _fjord_fast(p, fjord[slot_p])
        gi = GI[who]
        glob[gi["spots"]] = spots
        glob[gi["vik"]] = p.vikings_left
        sh = p.shields
        glob[gi["shields"]] = [1 if sh["extra"] else 0, 1 if sh["occupy"] else 0,
                               1 if sh["double"] else 0]
        glob[gi["axes"]] = sc["axes"]
        glob[gi["count"]] = sc["count"]
        glob[gi["value"]] = sc["value"]
        glob[gi["sites"]] = sc["sites"]
        glob[gi["unfilled"]] = sc["unfilled"]
        glob[gi["total"]] = sc["total"]
        glob[gi["room"]] = sc["room"]

    glob[G["to_move_me"]] = 1 if (g.current == me and not g.game_over) else 0
    for i, owner in enumerate(g.trophy_owner):
        if owner == me:
            glob[GI_TROPHY[i][0]] = 1
        elif owner == opp:
            glob[GI_TROPHY[i][1]] = 1
    glob[G["bag"]] = len(g.bag)
    for k, sid in enumerate(g.ocean_ships):
        if sid is None:
            continue
        need, cat, bonus = L.LONGSHIPS[sid]
        res_i, bonus_i = GI_OCEAN[k]
        for r in need:
            glob[res_i[r]] += 1
        glob[bonus_i[cat]] = bonus
    return land, fjord, glob
