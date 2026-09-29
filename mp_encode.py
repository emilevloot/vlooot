"""
The encoding for games of 2, 3 or 4 players ("mp" = multi-player).

It is the 2-player encoding (nn_encode.py, version 4) stretched to more
players, with the same features in the same places wherever possible, so a
trained 2-player network can hand over what it learned:

  land   [105, LAND_F]    all 4 landscape boards (100 spaces, boards 1-2 first,
                          in the same order as nn_encode) + the 5 ocean spaces.
                          Boards that are not in this game are "not in play".
                          Every "opp" feature now means the OPPONENTS: their
                          Vikings added up, and for chains, towers, castles and
                          "what a Viking here would give" the most dangerous
                          opponent (the largest number; for "Vikings needed
                          for the next castle tile" the smallest).
  fjord  [4, 37, FJORD_F] my fjord, then the other seats in playing order
                          after me (all zeros for seats not in the game);
                          exactly nn_encode's fjord features
  glob   [GLOB_F]         nn_encode's global numbers, where "opp" is the
                          STRONGEST opponent (highest score now), then extra
                          numbers per seat, the number of players and who is
                          in the game

Labels: margin = my final score - the best other score; win = 1 (shared:
1 / number of winners) or 0.

encode_reference() here is the plain version; mp_game.py has the fast one.
"""

import numpy as np

import looot as L
import nn_encode as E2

MAX_PLAYERS = 4
LAND_CELLS = [c for b in L.board_cells(MAX_PLAYERS) for c in b] + list(L.OCEAN_CELLS)
LAND_INDEX = {c: i for i, c in enumerate(LAND_CELLS)}
N_LAND = len(LAND_CELLS)                    # 105
N_BOARD = N_LAND - len(L.OCEAN_CELLS)       # 100
assert LAND_CELLS[:50] == E2.LAND_CELLS[:50]      # boards 1-2: the same spaces, same order
FJORD_CELLS, FJORD_INDEX, N_FJORD = E2.FJORD_CELLS, E2.FJORD_INDEX, E2.N_FJORD
LAND_NB = E2.neighbour_table(LAND_CELLS, LAND_INDEX)
FJORD_NB = E2.FJORD_NB

# the same land features as nn_encode, plus "in play"
TERRAINS, RES, BLD, CATS, SITE_ITEMS, GAIN = (E2.TERRAINS, E2.RES, E2.BLD, E2.CATS,
                                              E2.SITE_ITEMS, E2.GAIN)
for _n in ("LF_TERRAIN", "LF_OCEAN", "LF_SHIP", "LF_VIK", "LF_FREE", "LF_STACK", "LF_CHAIN",
           "LF_STOCK", "LF_TLINKS", "LF_TTOUCH", "LF_CLEVEL", "LF_CNEXT", "LF_GAIN", "LF_TURNS"):
    globals()[_n] = getattr(E2, _n)
for _n in dir(E2):
    if _n.startswith("FF_"):
        globals()[_n] = getattr(E2, _n)
LF_INPLAY = E2.LAND_F
LAND_F = LF_INPLAY + 1                      # 39
FJORD_F = E2.FJORD_F                        # 35
MAX_TURNS = E2.MAX_TURNS
ENCODING_VERSION = "mp1"

SEATS = ["next", "next2", "next3"]          # the other seats, in playing order after me
GLOB_NAMES = list(E2.GLOB_NAMES) + (
    ["players_%d" % n for n in (2, 3, 4)] +
    ["%s_%s" % (w, s) for s in SEATS for w in ("present", "total", "vik", "turns", "strongest")])
GLOB_F = len(GLOB_NAMES)
G = {n: i for i, n in enumerate(GLOB_NAMES)}
# the places a 2-player network's glob weights move to (same name, same meaning)
GLOB_FROM_2P = [G[n] for n in E2.GLOB_NAMES]

# for the networks (nn_model): the same kinds of columns as nn_encode
CLOCK_COLS = [G["turns_left_me"], G["turns_left_opp"]]
RES_COLS, GAIN_BLD = E2.RES_COLS, E2.GAIN_BLD
SHIP_NEED = [[G["ocean%d_%s" % (s, r)] for r in RES] for s in range(5)]
SHIP_BONUS = [[G["ocean%d_bonus_%s" % (s, it)] if it in CATS else 0 for it in SITE_ITEMS]
              for s in range(5)]
BONUS_ON = E2.BONUS_ON
PRESENT_COLS = [G["present_" + s] for s in SEATS]


def seats(g, me):
    """Player indices: me, then the others in playing order after me."""
    n = len(g.players)
    return [(me + k) % n for k in range(n)]


def strongest(g, me):
    """The opponent with the highest score now (first in playing order on a tie)."""
    others = seats(g, me)[1:]
    return max(others, key=lambda i: (g.players[i].score()["total"], -others.index(i)))


def encode_reference(g, me):
    """Encode game g (2-4 players) as seen by player index `me`.
    Returns (land int8 [105, LAND_F], fjord int8 [4, 37, FJORD_F], glob int16)."""
    order = seats(g, me)
    opps = order[1:]
    land = np.zeros((N_LAND, LAND_F), dtype=np.int8)
    fjord = np.zeros((MAX_PLAYERS, N_FJORD, FJORD_F), dtype=np.int8)
    glob = np.zeros(GLOB_F, dtype=np.int16)
    P = g.players

    # --- fjords (every seat), shopping lists ---
    needs, spots = {}, {}
    for k, idx in enumerate(order):
        needs[idx] = [0] * len(SITE_ITEMS)
        spots[idx] = E2._fjord_reference(P[idx], fjord[k], needs[idx])
    tl = {idx: E2.turns_left(P[idx]) for idx in order}
    for k, idx in enumerate(order):
        fjord[k, :, E2.FF_TURNS] = tl[idx]
        fjord[k, :, E2.FF_TURNS + 1] = max(tl[j] for j in order if j != idx)

    # --- landscape ---
    chains = {}
    for idx in order:
        chains[idx] = {}
        for c, owners in g.vikings.items():
            if idx in owners and c not in chains[idx]:
                ch = g.chain(P[idx], c)
                for cc in ch:
                    chains[idx][cc] = len(ch)
    links = {idx: {} for idx in order}
    for idx in order:
        for pk in P[idx].wt_pairs:
            for kk in pk.split("|"):
                links[idx][kk] = links[idx].get(kk, 0) + 1
    in_play = set(g.land)
    best = {"me": [0, 0], "opp": [0, 0]}
    for c in LAND_CELLS[:N_BOARD]:
        i = LAND_INDEX[c]
        if c not in in_play:
            continue
        t = g.land[c]
        land[i, LF_INPLAY] = 1
        land[i, LF_TERRAIN + TERRAINS.index(t)] = 1
        v = g.vikings.get(c, [])
        k = L.key(c)
        land[i, LF_VIK] = v.count(me)
        land[i, LF_VIK + 1] = min(127, sum(v.count(j) for j in opps))
        land[i, LF_CHAIN] = min(chains[me].get(c, 0), 127)
        land[i, LF_CHAIN + 1] = min(max(chains[j].get(c, 0) for j in opps), 127)
        if t == "watchtower":
            land[i, LF_TLINKS] = links[me].get(k, 0)
            land[i, LF_TLINKS + 1] = max(links[j].get(k, 0) for j in opps)
            touch = {j: 1 if any(j in g.vikings.get(n, []) for n in L.neighbours(c)) else 0
                     for j in order}
            land[i, LF_TTOUCH] = touch[me]
            land[i, LF_TTOUCH + 1] = max(touch[j] for j in opps)
        if t == "castle":
            nxt = {}
            for j in order:
                have = P[j].castle_taken.get(k, 0)
                nxt[j] = 0
                if g.rules["buildings"] and have < 3 and g.stock.get(c, 0) > 0:
                    biggest = max([chains[j].get(n, 0) for n in L.neighbours(c)])
                    nxt[j] = max(1, 4 * (have + 1) - biggest)
            land[i, LF_CLEVEL] = P[me].castle_taken.get(k, 0)
            land[i, LF_CLEVEL + 1] = max(P[j].castle_taken.get(k, 0) for j in opps)
            land[i, LF_CNEXT] = nxt[me]
            pos = [nxt[j] for j in opps if nxt[j] > 0]
            land[i, LF_CNEXT + 1] = min(pos) if pos else 0
        if t in BLD:
            land[i, LF_STOCK + BLD.index(t)] = g.stock.get(c, 0)
        if t in L.TERRAIN_RESOURCE and g.is_anchor(c):
            if v:
                land[i, LF_STACK] = 1
            else:
                land[i, LF_FREE] = 1
                glob[G["anchors_" + L.TERRAIN_RESOURCE[t]]] += 1
                gains = {j: E2._placement_gain(g, j, c, needs[j]) for j in order}
                opp_gain = [max(gains[j][x] for j in opps) for x in range(len(GAIN))]
                for x in range(len(GAIN)):
                    land[i, LF_GAIN + x] = min(gains[me][x], 127)
                    land[i, LF_GAIN + len(GAIN) + x] = min(opp_gain[x], 127)
                best["me"][0] = max(best["me"][0], 1 + sum(gains[me][1:4]))
                best["me"][1] = max(best["me"][1], gains[me][4])
                best["opp"][0] = max(best["opp"][0], max(1 + sum(gains[j][1:4]) for j in opps))
                best["opp"][1] = max(best["opp"][1], max(gains[j][4] for j in opps))
    for slot, c in enumerate(g.ocean):
        i = LAND_INDEX[c]
        land[i, LF_INPLAY] = 1
        land[i, LF_OCEAN] = 1
        land[i, LF_SHIP] = 1 if g.ocean_ships[slot] is not None else 0
    land[:, LF_TURNS] = tl[me]
    land[:, LF_TURNS + 1] = max(tl[j] for j in opps)

    # --- global: me and the strongest opponent (the 2-player names) ---
    top = strongest(g, me)
    glob[G["to_move_me"]] = 1 if (g.current == me and not g.game_over) else 0
    for who, idx in (("me", me), ("opp", top)):
        p = P[idx]
        glob[G["vik_" + who]] = p.vikings_left
        for sh in ("extra", "occupy", "double"):
            glob[G["shield_%s_%s" % (who, sh)]] = 1 if p.shields[sh] else 0
        glob[G["axes_" + who]] = g.axes(p)
        sc = p.score()
        for cat in CATS:
            glob[G["count_%s_%s" % (cat, who)]] = sc["rows"][cat]["count"]
            glob[G["value_%s_%s" % (cat, who)]] = sc["rows"][cat]["value"]
        glob[G["sites_" + who]] = sc["sites"]
        glob[G["unfilled_" + who]] = sc["unfilled"]
        glob[G["total_" + who]] = sc["total"]
        glob[G["room_" + who]] = len(p.empty_cells())
        glob[G["ship_spots_" + who]] = spots[idx]
        for t, nd in enumerate(needs[idx]):
            glob[G["need_%s_%s" % (SITE_ITEMS[t], who)]] = nd
        glob[G["turns_left_" + who]] = tl[idx]
        glob[G["turns_%s_%d" % (who, tl[idx])]] = 1
    for who in ("me", "opp"):
        glob[G["best_gain_" + who]] = best[who][0]
        glob[G["best_useful_" + who]] = best[who][1]
    for i, owner in enumerate(g.trophy_owner):
        if owner == me:
            glob[G["trophy%d_me" % i]] = 1
        elif owner is not None:
            glob[G["trophy%d_opp" % i]] = 1
    glob[G["bag"]] = len(g.bag)
    for k, sid in enumerate(g.ocean_ships):
        if sid is None:
            continue
        need, cat, bonus = L.LONGSHIPS[sid]
        for r in need:
            glob[G["ocean%d_%s" % (k, r)]] += 1
        glob[G["ocean%d_bonus_%s" % (k, cat)]] = bonus
    # --- the extra multi-player numbers ---
    glob[G["players_%d" % len(P)]] = 1
    for k, idx in enumerate(opps):
        s = SEATS[k]
        glob[G["present_" + s]] = 1
        glob[G["total_" + s]] = P[idx].score()["total"]
        glob[G["vik_" + s]] = P[idx].vikings_left
        glob[G["turns_" + s]] = tl[idx]
        glob[G["strongest_" + s]] = 1 if idx == top else 0
    return land, fjord, glob


def outcome(g, me):
    """(margin, win) for `me` once the game is over: my score minus the best
    other score, and my share of the win."""
    s = [p.score()["total"] for p in g.players]
    margin = s[me] - max(s[j] for j in range(len(s)) if j != me)
    w = g.winners()
    return float(margin), (1.0 / len(w)) if me in w else 0.0


def encode(g, me):
    """The encoding the players use: mp_game.py's compiled version (exactly
    the numbers of encode_reference, checked by test_mp.py)."""
    import mp_game                                   # (mp_game imports this file)
    s, gs = mp_game.from_game(g)
    return mp_game.encode_state(s, gs, me)
