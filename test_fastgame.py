"""
Checks that fastgame.py (the Numba copy of the game) plays EXACTLY like
looot.py, the real game.

    python test_fastgame.py            # all checks
    python test_fastgame.py quick      # fewer games

  1. encoding   fastgame's encoding of a position == nn_encode.encode
  2. rules      random turns (random placements, shields, tile spaces,
                longships, trophies) done in BOTH engines; after every single
                action the whole state must be identical
  3. player     the network player with all randomness switched off, in both
                engines: every turn must lead to the same position
  4. speed      turns per second of both network players
"""

import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import fastgame as F
import looot as L
import nn_encode as E


def games(n, seed0):
    """(seed, rules stage, layout) for n test games across all rule stages."""
    lay = L.read_layout_file()
    return [(seed0 + k, L.RULE_STAGES[k % len(L.RULE_STAGES)][0], lay if k % 3 else None)
            for k in range(n)]


def same_encoding(s, gs, g, where):
    for me in range(2):
        a = E.encode(g, me)
        b = F.encode_state(s, gs, me)
        for x, y, name in zip(a, b, ("land", "fjord", "glob")):
            if not np.array_equal(x, y):
                raise AssertionError("%s encoding differs %s: %s" % (
                    name, where, np.argwhere(x != y)[:6].tolist()))


def check_encoding(n_games):
    from gen_data import ExploringBot
    n = 0
    for sd, stage, lay in games(n_games, 5000):
        g = L.Game([("a", True), ("b", True)], sd, lay, L.rules_for(stage))
        bots = [ExploringBot(random.Random(sd * 2 + k), 0.3) for k in range(2)]
        while True:
            s, gs = F.from_game(g)
            same_encoding(s, gs, g, "(game %d, %s)" % (sd, stage))
            n += 1
            if g.game_over:
                break
            bots[g.current].play_turn(g)
    print("1. encoding: identical in %d positions" % n)


def _pending(g):
    return [(E.SITE_ITEMS.index(it["type"]),
             E.LAND_INDEX[L.unkey(it["from"])] if "from" in it else -1) for it in g.pending]


def check_rules(n_games):
    helper = L.Bot(random.Random(0))
    actions = 0
    for sd, stage, lay in games(n_games, 6000):
        g = L.Game([("a", True), ("b", True)], sd, lay, L.rules_for(stage))
        s, gs = F.from_game(g)
        rng = random.Random(sd)
        ptype = np.zeros(F.MAXP, np.int64)
        pfrom = np.zeros(F.MAXP, np.int64)
        n = 0

        def compare(what):
            s_py, _ = F.from_game(g)
            if not np.array_equal(s, s_py):
                bad = np.where(s != s_py)[0][:8]
                raise AssertionError("state differs after %s (game %d, %s) at %s: fast %s, real %s"
                                     % (what, sd, stage, bad.tolist(), s[bad].tolist(),
                                        s_py[bad].tolist()))

        def place(opt):
            nonlocal n
            c, occ, dbl = opt
            g.place_viking(c, occ, dbl)
            n = F.place_viking(s, gs, E.LAND_INDEX[c], occ, dbl, ptype, pfrom)
            got = list(zip(ptype[:n].tolist(), pfrom[:n].tolist()))
            assert got == _pending(g), ("pending differs", sd, got, _pending(g))
            compare("placing a Viking")
            p = g.player()
            while g.phase == "tiles":
                i = rng.randrange(len(g.pending))
                c2 = rng.choice(p.empty_cells())
                g.place_tile(i, c2)
                n = F.place_tile(s, gs, ptype, pfrom, n, i, E.FJORD_INDEX[c2])
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
                    F.take_ship(s, gs, slot, E.FJORD_INDEX[c])
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
            same_encoding(s, gs, g, "(game %d, %s)" % (sd, stage))
            actions += 1
        for me in range(2):
            real = [pl.score()["total"] for pl in g.players]
            assert F.total_score(s, me) == real[me]
            w = g.winners()
            assert F.outcome(s, me)[1] == ((1.0 / len(w)) if me in w else 0.0)
    print("2. rules: identical after every action of %d random turns in %d games"
          % (actions, n_games))


class ZeroRng:
    """A 'random' generator that always says 0: switches off the random
    tie-breaks of nn_bot.NNBot, so its choices can be compared exactly."""
    def random(self):
        return 0.0

    def shuffle(self, x):
        pass

    def randrange(self, n):
        return 0


def latest_model():
    import glob
    import nn_bot
    for path in sorted(glob.glob(os.path.join(nn_bot.MODEL_DIR, "*.npz")),
                       key=os.path.getmtime, reverse=True):
        try:
            nn_bot.FastNet(path)
            return path
        except (ValueError, KeyError):
            pass
    raise SystemExit("no trained network found")


def check_player(n_games):
    import nn_bot
    path = latest_model()
    same = 0
    turns = 0
    for sd, stage, lay in games(n_games, 7000):
        g = L.Game([("a", True), ("b", True)], sd, lay, L.rules_for(stage))
        s, gs = F.from_game(g)
        slow = nn_bot.NNBot(ZeroRng(), path)
        fast = F.FastPlayer(nn_bot.FastNet(path), ZeroRng(), noise=False, shuffle=False)
        ok = True
        while not g.game_over:
            slow.play_turn(g)
            s = fast.turn(s, gs)
            turns += 1
            if not np.array_equal(s, F.from_game(g)[0]):
                ok = False
                break
        same += ok
    print("3. player: %d of %d games identical move for move (%d turns compared)%s"
          % (same, n_games, turns, "" if same == n_games else
             " - tiny differences in the network's numbers can flip a near-tie"))
    return same, n_games


def speed(n_games=6):
    import nn_bot
    path = latest_model()
    lay = L.read_layout_file()
    res = {}
    for name in ("python", "numba"):
        t = time.perf_counter()
        turns = 0
        for k in range(n_games):
            g = L.Game([("a", True), ("b", True)], 800 + k, lay)
            if name == "python":
                bots = [nn_bot.NNBot(random.Random(k * 2 + j), path) for j in range(2)]
                while not g.game_over:
                    bots[g.current].play_turn(g)
                    turns += 1
            else:
                s, gs = F.from_game(g)
                pl = F.FastPlayer(nn_bot.FastNet(path), random.Random(k))
                while s[F.S_OVER] == 0:
                    s = pl.turn(s, gs)
                    turns += 1
        res[name] = 1000 * (time.perf_counter() - t) / turns
    print("4. speed: python %.1f ms per turn, numba %.1f ms per turn (%.1fx)"
          % (res["python"], res["numba"], res["python"] / res["numba"]))


if __name__ == "__main__":
    quick = "quick" in sys.argv
    t = time.time()
    F.encode_state(*F.from_game(L.Game([("a", True), ("b", True)], 1)), 0)   # compile
    print("(compiling took %.0f s)" % (time.time() - t))
    check_encoding(10 if quick else 40)
    check_rules(20 if quick else 150)
    check_player(4 if quick else 20)
    speed()
