"""Build mp_game.py (up to 4 players) from fastgame.py (2 players)."""
import re

s = open("fastgame.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, b)


# docstring
i = s.index('"""', 3)
s = '''"""
Looot for 2, 3 or 4 players as numpy arrays, compiled with Numba: the
multi-player copy of fastgame.py (same rules, same network player), with
the encoding of mp_encode.py. Up to 4 players in one state array: every
per-player block has room for MAXPL = 4; gs[G_NP] says how many play.
Spaces of landscape boards that are not in the game have terrain T_OUT.

test_mp.py checks that it plays exactly like looot.py.
''' + s[i:]

sub("import nn_encode as E", "import mp_encode as E")
sub("MAXO = 160           # placement options in one turn", "MAXO = 400           # placement options in one turn")
sub("T_HOUSE, T_TOWER, T_CASTLE, T_OCEAN = 4, 5, 6, 7", "T_HOUSE, T_TOWER, T_CASTLE, T_OCEAN, T_OUT = 4, 5, 6, 7, 8   # T_OUT: board not in this game")

# state layout
a = s.index("S_VIK = 0 ")
b = s.index("S_OVER = S_CUR + 1")
s = s[:a] + '''MAXPL = E.MAX_PLAYERS
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
''' + s[b:]
sub("G_SITENEED = G_NT + 1           # [2][3][7] what each construction site needs\n"
    "G_RULES = G_SITENEED + 2 * 3 * 7   # [5] buildings, sites, longships, shields, trophies\n"
    "G_LEN = G_RULES + 5",
    "G_SITENEED = G_NT + 1           # [MAXPL][3][7] what each construction site needs\n"
    "G_RULES = G_SITENEED + MAXPL * 3 * 7   # [5] buildings, sites, longships, shields, trophies\n"
    "G_NP = G_RULES + 5              # number of players\n"
    "G_LEN = G_NP + 1")

# encoding layout names
sub("LF_GAIN, N_GAIN, LF_TURNS = E.LF_GAIN, len(E.GAIN), E.LF_TURNS",
    "LF_GAIN, N_GAIN, LF_TURNS, LF_INPLAY = E.LF_GAIN, len(E.GAIN), E.LF_TURNS, E.LF_INPLAY")
sub("GI_TOMOVE, GI_BAG = G_[\"to_move_me\"], G_[\"bag\"]",
    "GI_TOMOVE, GI_BAG = G_[\"to_move_me\"], G_[\"bag\"]\n"
    "GI_PLAYERS = np.array([G_[\"players_%d\" % n] for n in (2, 3, 4)])\n"
    "GI_SEAT = np.array([[G_[\"%s_%s\" % (w, s_)] for w in (\"present\", \"total\", \"vik\", \"turns\", \"strongest\")]\n"
    "                    for s_ in E.SEATS])")
sub('stamp = os.path.join(cache, "fastgame.sig")', 'stamp = os.path.join(cache, "mp_game.sig")')
sub('glob.glob(os.path.join(cache, "fastgame.*.nb[ic]"))', 'glob.glob(os.path.join(cache, "mp_game.*.nb[ic]"))')

# from_game
sub('    assert len(g.players) == 2, "fastgame is for 2-player games"',
    '    assert 2 <= len(g.players) <= MAXPL')
sub("""    for i in range(NB):
        code = E.TERRAINS.index(g.land[E.LAND_CELLS[i]])""",
    """    gs[G_NP] = len(g.players)
    for i in range(NB):
        cell = E.LAND_CELLS[i]
        if cell not in g.land:
            gs[G_TERR + i] = T_OUT
            continue
        code = E.TERRAINS.index(g.land[cell])""")
sub("""        s[S_VIK + i] = v.count(0)
        s[S_VIK + NL + i] = v.count(1)""", """        for p in range(len(g.players)):
            s[S_VIK + p * NL + i] = v.count(p)""")

# who is on a space
sub("""@njit(cache=True)
def is_anchor(s, gs, i):""", """@njit(cache=True)
def occupied(s, gs, i):
    \"\"\"Any player's Viking on space i.\"\"\"
    for p in range(gs[G_NP]):
        if s[S_VIK + p * NL + i] > 0:
            return True
    return False


@njit(cache=True)
def is_anchor(s, gs, i):""")
sub("        elif s[S_VIK + n] + s[S_VIK + NL + n] > 0:\n            return True",
    "        elif occupied(s, gs, n):\n            return True")
sub("        occ = s[S_VIK + i] + s[S_VIK + NL + i] > 0", "        occ = occupied(s, gs, i)")

# turn order
sub("""    cur = s[S_CUR]
    for step in range(1, 3):
        nxt = (cur + step) % 2""", """    cur = s[S_CUR]
    npl = gs[G_NP]
    for step in range(1, npl + 1):
        nxt = (cur + step) % npl""")

# the result
a = s.index("@njit(cache=True)\ndef outcome(s, me):")
b = s.index("# ---------------------------------------------------------------------------", a)
s = s[:a] + '''@njit(cache=True)
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


''' + s[b:]

# the encoding
a = s.index("# Encoding: exactly the numbers of nn_encode.encode_reference")
b = s.index("@njit(cache=True)\ndef encode_many")
s = s[:a] + open("mp_enc_block.py", encoding="utf-8").read() + s[b:]
s = s.replace("2, NF, FJORD_F", "MAXPL, NF, FJORD_F")

# outcome callers
sub("            v[k] = outcome(states[k], me)[0]", "            v[k] = outcome(states[k], gs, me)[0]")

# training data: one row per player per turn
a = s.index("def record(states, gs, seed_")
b = s.index("def selfplay_game(")
s = s[:a] + '''def record(states, gs, seed_, targets=None):
    """Training data of a finished game: every turn-start position seen by
    every player, labelled with how the game ended for that player. Rows:
    turn 0 (player 0, 1, ...), turn 1 (player 0, 1, ...), ... - the format
    of gen_data.play_and_record. With `targets` (one per turn, from
    FastPlayer.target) also what the proposals should learn, on the row of
    the player whose turn it was (pol_on = 1 there)."""
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
    if targets is None:
        return land, fjord, glob, margin, win, game
    pp = np.zeros((npl * n, NL, 3), np.float16)
    pt = np.zeros((npl * n, NF, 8), np.float16)
    ps = np.zeros((npl * n, 6), np.float16)
    on = np.zeros(npl * n, np.int8)
    for t, tg in enumerate(targets):
        if tg is None:
            continue
        r = npl * t + int(states[t][S_CUR])
        pp[r], pt[r], ps[r] = tg
        on[r] = 1
    return land, fjord, glob, margin, win, game, pp, pt, ps, on


''' + s[b:]
sub("def selfplay_game(seed_, layout, rules, net, explore=0.0, noise=True):",
    "def selfplay_game(seed_, layout, rules, net, explore=0.0, noise=True, players=4):")
sub("def selfplay_batched(seeds, layout, rules, net, explore=0.0, parallel=64, noise=True):",
    "def selfplay_batched(seeds, layout, rules, net, explore=0.0, parallel=64, noise=True, players=4):")
sub('L.Game([("p0", True), ("p1", True)], ', 'L.Game([("p%d" % i, True) for i in range(players)], ', n=2)
sub("        assert len(states) < 200", "        assert len(states) < 400")

sub("            if s0[S_VIK + i] + s0[S_VIK + NL + i] > 0:",
    "            if _any_viking(s0, i):")
sub("@njit(cache=True)\ndef policy_target(s0, s1, w, place, tile, ship):",
    "@njit(cache=True)\ndef _any_viking(s, i):\n"
    "    \"\"\"A Viking of any player on space i.\"\"\"\n"
    "    for p in range(MAXPL):\n"
    "        if s[S_VIK + p * NL + i] > 0:\n"
    "            return True\n"
    "    return False\n\n\n"
    "@njit(cache=True)\ndef policy_target(s0, s1, w, place, tile, ship):")
open("mp_game.py", "w", encoding="utf-8", newline="\n").write(s)
left = [m.start() for m in re.finditer(r"S_VIK \+ NL|2 \* NL|range\(2\)", s)]
print("mp_game.py written; places to look at:", [s.count("\n", 0, p) + 1 for p in left])
