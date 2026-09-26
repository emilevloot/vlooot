"""
Checks that every fast version gives exactly the results of the simple one.

    python test_speedups.py

  1. nn_encode.encode()  == nn_encode.encode_reference()   (every number)
  2. nn_bot.FastNet      == nn_bot.NumpyNet                (to 1e-4 points)
  3. the rules still pass the scenario tests (tower groups, stacks, ...)
"""

import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import looot as L
import nn_encode as E


def check_encoding(games=40):
    """Positions from games with mixed players under every rule stage."""
    from gen_data import ExploringBot
    lay = L.read_layout_file()
    n = 0
    for s in range(games):
        stage = L.RULE_STAGES[s % len(L.RULE_STAGES)][0]
        g = L.Game([("a", True), ("b", True)], 4000 + s, lay if s % 3 else None,
                   L.rules_for(stage))
        bots = [ExploringBot(random.Random(s * 2 + k), 0.3) for k in range(2)]
        while True:
            for me in range(2):
                a, b = E.encode_reference(g, me), E.encode(g, me)
                for x, y, name in zip(a, b, ("land", "fjord", "glob")):
                    if x.dtype != y.dtype or not np.array_equal(x, y):
                        raise AssertionError("%s differs (stage %s, game %d): %s" % (
                            name, stage, s, np.argwhere(x != y)[:5].tolist()))
                n += 1
            if g.game_over:
                break
            bots[g.current].play_turn(g)
    print("1. encoding identical on %d positions" % n)


def check_net(model=None):
    import glob
    import nn_bot
    models = sorted(glob.glob(os.path.join(nn_bot.MODEL_DIR, "*.npz")), key=os.path.getmtime)
    path = model or next(m for m in reversed(models) if _loads(nn_bot, m))
    slow, fast = nn_bot.NumpyNet(path), nn_bot.FastNet(path)
    g = L.Game([("a", True), ("b", True)], 11, L.read_layout_file())
    bot = nn_bot.NNBot(random.Random(1), path)
    batches = []
    real = bot.net
    bot.net = lambda l, f, gl: (batches.append((l, f, gl)), real(l, f, gl))[1]
    while not g.game_over and len(batches) < 60:
        bot.play_turn(g)
    worst = 0.0
    tiny = nn_bot.FastNet(path)
    tiny.CACHE_SIZE = 40              # a cache that is full all the time
    for l, f, gl in batches:
        m1, w1 = slow(l, f, gl)
        for net in (fast, tiny):
            m2, w2 = net(l, f, gl)
            worst = max(worst, float(np.abs(m1 - m2).max()), float(np.abs(w1 - w2).max()))
    assert worst < 1e-4, worst
    print("2. network identical on %d positions (largest difference %.1g) - %s" % (
        sum(len(b[0]) for b in batches), worst, os.path.basename(path)))


def _loads(nn_bot, path):
    try:
        nn_bot.FastNet(path)
        return True
    except (ValueError, KeyError):
        return False


def check_rules():
    lay = L.read_layout_file()
    g = L.Game([("a", True), ("b", True)], 1, lay)
    towers = [c for c, t in g.land.items() if t == "watchtower"][:3]
    X, T2, T1 = towers
    p = g.players[0]
    assert len(g.tower_tiles(p, [X, T2])) == 2          # two new towers
    assert len(g.tower_tiles(p, [X, T2])) == 0          # nothing new
    pay = g.tower_tiles(p, [X, T2, T1])                  # the Ragnar example
    assert len(pay) == 2 and T1 in pay
    house = next(c for c, t in g.land.items() if t == "house")
    assert [len(g.take_building(house, 1)) for _ in range(3)] == [1, 1, 0]
    for n in (2, 3, 4):
        for seed in range(4):
            gg = L.simulate(n, seed, verbose=False, layout=lay if seed % 2 else None)
            assert gg.game_over and all(v >= 0 for v in gg.stock.values())
    print("3. rules scenarios and 24 full games ok")


if __name__ == "__main__":
    check_rules()
    check_encoding()
    check_net()
