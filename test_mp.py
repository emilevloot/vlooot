"""
Checks that mp_game.py (the Numba copy of the game for 2-4 players) plays
EXACTLY like looot.py, and encodes exactly like mp_encode.encode_reference.

    python test_mp.py            # all checks
    python test_mp.py quick      # fewer games

  1. encoding   mp_game's encoding == mp_encode.encode_reference, and for
                2 players the same numbers as the 2-player encoding
  2. rules      random turns in BOTH engines with 2, 3 and 4 players; after
                every single action the whole state must be identical
  3. player     the network player with all randomness switched off, in both
                engines (needs a trained multi-player network)
"""

import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import looot as L
import mp_encode as M
import mp_game as F


def games(n, seed0):
    """(seed, players, rules stage, layout) for n test games."""
    lay = L.read_layout_file()
    return [(seed0 + k, 2 + k % 3, L.RULE_STAGES[k % len(L.RULE_STAGES)][0], lay if k % 4 else None)
            for k in range(n)]


def new_game(sd, npl, stage, lay):
    return L.Game([("p%d" % i, True) for i in range(npl)], sd, lay, L.rules_for(stage))


def same_encoding(s, gs, g, where):
    for me in range(len(g.players)):
        a = M.encode_reference(g, me)
        b = F.encode_state(s, gs, me)
        for x, y, name in zip(a, b, ("land", "fjord", "glob")):
            if x.dtype != y.dtype or not np.array_equal(x, y):
                bad = np.argwhere(x != y)[:6]
                raise AssertionError("%s encoding differs %s (me %d): %s ref %s fast %s" % (
                    name, where, me, bad.tolist(), [x[tuple(i)] for i in bad], [y[tuple(i)] for i in bad]))


def check_encoding(n_games):
    from gen_data import ExploringBot
    n = 0
    for sd, npl, stage, lay in games(n_games, 5000):
        g = new_game(sd, npl, stage, lay)
        bots = [ExploringBot(random.Random(sd * 5 + k), 0.3) for k in range(npl)]
        while True:
            s, gs = F.from_game(g)
            same_encoding(s, gs, g, "(game %d, %d players, %s)" % (sd, npl, stage))
            n += len(g.players)
            if g.game_over:
                break
            bots[g.current].play_turn(g)
    print("1. encoding: identical in %d positions (2, 3 and 4 players)" % n)


def _pending(g):
    return [(M.SITE_ITEMS.index(it["type"]),
             M.LAND_INDEX[L.unkey(it["from"])] if "from" in it else -1) for it in g.pending]


def check_rules(n_games):
    helper = L.Bot(random.Random(0))
    actions = 0
    for sd, npl, stage, lay in games(n_games, 6000):
        g = new_game(sd, npl, stage, lay)
        s, gs = F.from_game(g)
        rng = random.Random(sd)
        ptype = np.zeros(F.MAXP, np.int64)
        pfrom = np.zeros(F.MAXP, np.int64)
        n = 0

        def compare(what):
            s_py, _ = F.from_game(g)
            if not np.array_equal(s, s_py):
                bad = np.where(s != s_py)[0][:8]
                raise AssertionError("state differs after %s (game %d, %d players, %s) at %s: fast %s, real %s"
                                     % (what, sd, npl, stage, bad.tolist(), s[bad].tolist(),
                                        s_py[bad].tolist()))

        def place(opt):
            nonlocal n
            c, occ, dbl = opt
            g.place_viking(c, occ, dbl)
            n = F.place_viking(s, gs, M.LAND_INDEX[c], occ, dbl, ptype, pfrom)
            got = list(zip(ptype[:n].tolist(), pfrom[:n].tolist()))
            assert got == _pending(g), ("pending differs", sd, got, _pending(g))
            compare("placing a Viking")
            p = g.player()
            while g.phase == "tiles":
                i = rng.randrange(len(g.pending))
                c2 = rng.choice(p.empty_cells())
                g.place_tile(i, c2)
                n = F.place_tile(s, gs, ptype, pfrom, n, i, M.FJORD_INDEX[c2])
                compare("placing a tile")

        while not g.game_over:
            p = g.player()
            if g.phase == "place":
                place(rng.choice(helper.placement_options(g)))
                if g.can_use_extra(p) != F.can_use_extra(s, gs):
                    raise AssertionError("can_use_extra differs (game %d)" % sd)
                if g.can_use_extra(p) and rng.random() < 0.6:
                    g.use_extra_shield()
                    F.use_extra(s)
                    compare("using the extra shield")
                    place(rng.choice(helper.placement_options(g)))
            if not g.took_ship and p.empty_cells() and rng.random() < 0.5:
                slots = [k for k, sid in enumerate(g.ocean_ships) if sid is not None]
                if slots:
                    slot, c = rng.choice(slots), rng.choice(p.empty_cells())
                    g.pick_ship(slot)
                    g.place_ship(c)
                    F.take_ship(s, gs, slot, M.FJORD_INDEX[c])
                    compare("taking a longship")
            cl = g.claimable_trophies(p)
            best = max(cl) if cl else -1
            if best != F.best_claimable(s, gs):
                raise AssertionError("claimable trophies differ (game %d)" % sd)
            if cl and rng.random() < 0.7:
                tr = rng.choice(cl)
                g.claim_trophy(tr)
                F.claim(s, tr)
                compare("claiming a trophy")
            g.end_turn()
            F.end_turn(s, gs)
            compare("ending the turn")
            if actions % 3 == 0:
                same_encoding(s, gs, g, "(game %d, %d players, %s)" % (sd, npl, stage))
            actions += 1
        for me in range(npl):
            assert F.outcome(s, gs, me) == M.outcome(g, me), (F.outcome(s, gs, me), M.outcome(g, me))
    print("2. rules: identical after every action of %d random turns in %d games (2-4 players)"
          % (actions, n_games))


class ZeroRng:
    """Switches off the random tie-breaks of the players."""
    def random(self):
        return 0.0

    def shuffle(self, x):
        pass

    def randrange(self, n):
        return 0


def check_player(n_games, model):
    import nn_bot
    same = turns = 0
    for sd, npl, stage, lay in games(n_games, 7000):
        stage = "full"
        g = new_game(sd, npl, stage, lay)
        s, gs = F.from_game(g)
        slow = nn_bot.NNBot(ZeroRng(), model)
        fast = F.FastPlayer(nn_bot.load_net(model), ZeroRng(), noise=False, shuffle=False)
        ok = True
        while not g.game_over:
            slow.play_turn(g)
            s = fast.turn(s, gs)
            turns += 1
            if not np.array_equal(s, F.from_game(g)[0]):
                ok = False
                break
        same += ok
    print("3. player: %d of %d games identical move for move (%d turns compared)" % (same, n_games, turns))


if __name__ == "__main__":
    quick = "quick" in sys.argv
    t = time.time()
    F.encode_state(*F.from_game(new_game(1, 4, "full", None)), 0)
    print("(compiling took %.0f s)" % (time.time() - t))
    check_encoding(9 if quick else 30)
    check_rules(15 if quick else 90)
    model = next((a for a in sys.argv[1:] if a != "quick"), None)
    if model:
        check_player(6 if quick else 15, model)
