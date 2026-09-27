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

Version 3 describes the land the way a player looks at it:
  - what would I get if I put a Viking HERE (resource, house, tower and
    castle tiles, how big my chain becomes) - and the same for the opponent
  - how close I am to the next tile of each castle and tower
  - which of those tiles are on my "shopping list": the items my unfinished
    longships and construction sites still miss (and the opponent's list)

encode_reference() below is the straightforward version. The players use
encode(), which runs the compiled copy in fastgame.py; test_speedups.py and
test_fastgame.py check that both give exactly the same numbers.
"""

import copy

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

# land features (pairs are always: me, then the opponent)
LF_TERRAIN = 0                  # 7 one-hot
LF_OCEAN = 7
LF_SHIP = 8                     # ocean space with a longship on it
LF_VIK = 9                      # 2: number of Vikings
LF_FREE = 11                    # free resource space where a Viking may go now
LF_STACK = 12                   # occupied resource space the occupy shield could use
LF_CHAIN = 13                   # 2: size of the chain this space is in
LF_STOCK = 15                   # 3: tiles left on this house / watchtower / castle
LF_TLINKS = 18                  # 2: watchtowers this watchtower is linked to
LF_TTOUCH = 20                  # 2: the player has a Viking next to this watchtower
LF_CLEVEL = 22                  # 2: tiles the player took from this castle
LF_CNEXT = 24                   # 2: Vikings the player's chain still needs for the
                                #    next tile of this castle (0: nothing to gain)
LF_GAIN = 26                    # 2 x 5: if the player put a Viking on this free space:
GAIN = ["chain", "house", "watchtower", "castle", "useful"]
                                #    chain size, tiles of each building, and how many
                                #    of the items gained are on the player's shopping list
LAND_F = LF_GAIN + 2 * len(GAIN)          # 36

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

ENCODING_VERSION = 3

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
              ["ship_spots_me", "ship_spots_opp"] +
              # version 3: the shopping list (items the longships and sites
              # that can still be finished miss), and the best single Viking
              # placement right now: most tiles, most useful tiles
              ["need_%s_%s" % (t, p) for p in ("me", "opp") for t in SITE_ITEMS] +
              ["best_gain_me", "best_gain_opp", "best_useful_me", "best_useful_opp"])
GLOB_F = len(GLOB_NAMES)
G = {n: i for i, n in enumerate(GLOB_NAMES)}


def _missing(need, have):
    miss = list(need)
    for h in have:
        if h in miss:
            miss.remove(h)
    return miss


def _empty_neighbours(p, c):
    return sum(1 for n in L.neighbours(c) if n in FJORD_INDEX and n not in p.fjord)


def _fjord_reference(p, out, need):
    """Fill out [N_FJORD, FJORD_F] for player p, and add the items still
    missing on ships/sites that can be finished to need[7] (SITE_ITEMS order).
    Returns the number of empty spaces where a new longship could still be filled."""
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
        else:
            continue
        if out[i, FF_FILLABLE]:
            for m in miss:
                need[SITE_ITEMS.index(m)] += 1
    return ship_spots


def _placement_gain(g, idx, c, need):
    """What player idx would get from a Viking on free space c (without
    shields): [chain size, house, watchtower and castle tiles, useful items].
    Done the plain way: capture on a copy of the game."""
    g2 = copy.deepcopy(g)
    p2 = g2.players[idx]
    chain = len(g2.chain(p2, c))
    kinds = [g.land[a] for a in g2.captures(p2, c)]
    got = [kinds.count(b) for b in BLD]
    res = RES.index(L.TERRAIN_RESOURCE[g.land[c]])
    useful = min(1, need[res]) + sum(min(n, need[4 + k]) for k, n in enumerate(got))
    return [chain] + got + [useful]


def encode_reference(g, me):
    """The straightforward version of encode(), kept to test the fast one.
    Encode game g as seen by player index `me`.
    Returns (land int8, fjord int8, glob int16)."""
    assert len(g.players) == N_PLAYERS
    order = (me, 1 - me)                       # w = 0: me, w = 1: the opponent
    land = np.zeros((N_LAND, LAND_F), dtype=np.int8)
    fjord = np.zeros((2, N_FJORD, FJORD_F), dtype=np.int8)
    glob = np.zeros(GLOB_F, dtype=np.int16)
    who = ("me", "opp")

    # --- fjords first: the shopping lists are needed for the land ---
    needs = []
    for w, idx in enumerate(order):
        need = [0] * len(SITE_ITEMS)
        glob[G["ship_spots_" + who[w]]] = _fjord_reference(g.players[idx], fjord[w], need)
        for t, n in enumerate(need):
            glob[G["need_%s_%s" % (SITE_ITEMS[t], who[w])]] = n
        needs.append(need)

    # --- landscape and ocean ---
    chains = [{} for _ in order]
    for w, idx in enumerate(order):
        p = g.players[idx]
        for c, owners in g.vikings.items():
            if idx in owners and c not in chains[w]:
                ch = g.chain(p, c)
                for cc in ch:
                    chains[w][cc] = len(ch)
    links = [{} for _ in order]
    for w, idx in enumerate(order):
        for pk in g.players[idx].wt_pairs:
            for k in pk.split("|"):
                links[w][k] = links[w].get(k, 0) + 1
    best = [[0, 0], [0, 0]]                    # [w] -> best gain, best useful
    for c, t in g.land.items():
        i = LAND_INDEX[c]
        land[i, LF_TERRAIN + TERRAINS.index(t)] = 1
        v = g.vikings.get(c, [])
        k = L.key(c)
        for w, idx in enumerate(order):
            p = g.players[idx]
            land[i, LF_VIK + w] = v.count(idx)
            land[i, LF_CHAIN + w] = min(chains[w].get(c, 0), 127)
            if t == "watchtower":
                land[i, LF_TLINKS + w] = links[w].get(k, 0)
                land[i, LF_TTOUCH + w] = 1 if any(
                    idx in g.vikings.get(n, []) for n in L.neighbours(c)) else 0
            if t == "castle":
                have = p.castle_taken.get(k, 0)
                land[i, LF_CLEVEL + w] = have
                if g.rules["buildings"] and have < 3 and g.stock.get(c, 0) > 0:
                    biggest = max([chains[w].get(n, 0) for n in L.neighbours(c)])
                    land[i, LF_CNEXT + w] = max(1, 4 * (have + 1) - biggest)
        if t in BLD:
            land[i, LF_STOCK + BLD.index(t)] = g.stock.get(c, 0)
        if t in L.TERRAIN_RESOURCE and g.is_anchor(c):
            if v:
                land[i, LF_STACK] = 1
            else:
                land[i, LF_FREE] = 1
                glob[G["anchors_" + L.TERRAIN_RESOURCE[t]]] += 1
                for w, idx in enumerate(order):
                    gain = _placement_gain(g, idx, c, needs[w])
                    for j, x in enumerate(gain):
                        land[i, LF_GAIN + w * len(GAIN) + j] = min(x, 127)
                    best[w][0] = max(best[w][0], 1 + sum(gain[1:4]))
                    best[w][1] = max(best[w][1], gain[4])
    for slot, c in enumerate(g.ocean):
        i = LAND_INDEX[c]
        land[i, LF_OCEAN] = 1
        land[i, LF_SHIP] = 1 if g.ocean_ships[slot] is not None else 0

    # --- everything else ---
    glob[G["to_move_me"]] = 1 if (g.current == me and not g.game_over) else 0
    for w, idx in enumerate(order):
        p, s = g.players[idx], who[w]
        glob[G["vik_" + s]] = p.vikings_left
        for sh in ("extra", "occupy", "double"):
            glob[G["shield_%s_%s" % (s, sh)]] = 1 if p.shields[sh] else 0
        glob[G["axes_" + s]] = g.axes(p)
        sc = p.score()
        for cat in CATS:
            glob[G["count_%s_%s" % (cat, s)]] = sc["rows"][cat]["count"]
            glob[G["value_%s_%s" % (cat, s)]] = sc["rows"][cat]["value"]
        glob[G["sites_" + s]] = sc["sites"]
        glob[G["unfilled_" + s]] = sc["unfilled"]
        glob[G["total_" + s]] = sc["total"]
        glob[G["room_" + s]] = len(p.empty_cells())
        glob[G["best_gain_" + s]] = best[w][0]
        glob[G["best_useful_" + s]] = best[w][1]
    for i, owner in enumerate(g.trophy_owner):
        if owner == me:
            glob[G["trophy%d_me" % i]] = 1
        elif owner == 1 - me:
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


def encode(g, me):
    """Encode game g as seen by player index `me`: (land int8, fjord int8,
    glob int16). The compiled version in fastgame.py; exactly the numbers
    of encode_reference(), many times faster."""
    import fastgame                            # (fastgame imports this file)
    s, gs = fastgame.from_game(g)
    return fastgame.encode_state(s, gs, me)


def outcome(g, me):
    """Training labels once the game is over: (margin, win) for `me`."""
    s = [p.score()["total"] for p in g.players]
    margin = s[me] - max(s[j] for j in range(len(s)) if j != me)
    w = g.winners()
    win = (1.0 / len(w)) if me in w else 0.0
    return float(margin), float(win)
