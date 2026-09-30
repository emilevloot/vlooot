"""
Play games between the network and the greedy player and record how the
network plays, for the dashboard (make_dashboard.py).

    python tools/dashboard_data.py --games 300 --model s2_r19

Per game: the network's win chance and expected margin at the start of every
turn (from its own seat), what each player took each turn (resources,
building tiles, longships, shields, trophy), the values the network gives
each item at the start of its turns (nn_bot.thoughts), and the final score
of both players per category. Written to results/dashboard_data.json.
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor

HERE = ROOT
ITEMS = ["wood", "sheep", "gold", "axe", "house", "watchtower", "castle"]
CATS = ["castle", "watchtower", "house", "gold", "sheep", "wood"]


def counts(p):
    """Items on player p's fjord, per item type."""
    c = dict.fromkeys(ITEMS, 0)
    for it in p.fjord.values():
        if it["kind"] in ("res", "bld"):
            c[it["type"]] += 1
    return c


def ships(p):
    return {k: dict(it) for k, it in p.fjord.items() if it["kind"] == "ship"}


def play(args):
    seed, model, nn_seat, with_values = args
    import looot as L
    import nn_bot
    lay = L.read_layout_file()
    g = L.Game([("p0", True), ("p1", True)], seed, lay)
    bots = [None, None]
    bots[nn_seat] = nn_bot.NNBot(random.Random(seed), model)
    bots[1 - nn_seat] = L.Bot(random.Random(seed + 1))
    net = bots[nn_seat].net
    enc = bots[nn_seat].enc
    turns = []
    t = 0
    while not g.game_over:
        me = g.current
        p = g.players[me]
        land, fjord, glob = enc.encode(g, nn_seat)
        margin, win = net(land[None], fjord[None], glob[None])
        row = {"t": t, "who": "nn" if me == nn_seat else "greedy",
               "vikings_left": p.vikings_left,
               "win": round(float(win[0]), 4), "margin": round(float(margin[0]), 2)}
        if me == nn_seat and with_values:
            th = nn_bot.thoughts(net, g, nn_seat)
            row["values"] = th["values"]
        before, ships0 = counts(p), ships(p)
        vik0 = {c: list(v) for c, v in g.vikings.items()}
        shields0 = dict(p.shields)
        trophy0 = p.trophy
        bots[me].play_turn(g)
        after, ships1 = counts(p), ships(p)
        row["got"] = {k: after[k] - before[k] for k in ITEMS if after[k] != before[k]}
        placed = [c for c, v in g.vikings.items() if v.count(me) > vik0.get(c, []).count(me)]
        row["terrain"] = [g.land[c] for c in placed]
        row["shields"] = [k for k in shields0 if shields0[k] and not p.shields[k]]
        new = [it for k, it in ships1.items() if k not in ships0]
        row["ship"] = [{"cat": it["cat"], "bonus": it["bonus"]} for it in new]
        if p.trophy is not None and trophy0 is None:
            row["trophy"] = L.TROPHIES[p.trophy][1]
        turns.append(row)
        t += 1
    final = {}
    for who, idx in (("nn", nn_seat), ("greedy", 1 - nn_seat)):
        sc = g.players[idx].score()
        final[who] = {"total": sc["total"], "sites": sc["sites"], "trophy": sc["trophy"],
                      "penalty": sc["penalty"],
                      **{c: sc["rows"][c]["points"] for c in CATS},
                      "filled": sum(1 for it in g.players[idx].fjord.values()
                                    if it["kind"] == "ship" and it["filled"]),
                      "unfilled": sc["unfilled"]}
    w = g.winners()
    return {"seed": seed, "nn_seat": nn_seat, "nn_won": (1.0 / len(w)) if nn_seat in w else 0.0,
            "turns": turns, "final": final}


def main():
    ap = argparse.ArgumentParser(description="Record games for the dashboard.")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--model", default="s2_r19")
    ap.add_argument("--values-every", type=int, default=2,
                    help="ask the network's item values in every Nth game (they cost time)")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(31_000_000 + i, a.model, i % 2, i % a.values_every == 0) for i in range(a.games)]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        games = list(ex.map(play, jobs, chunksize=2))
    out = {"model": a.model, "opponent": "greedy", "made": time.strftime("%Y-%m-%d %H:%M"),
           "games": games}
    with open(os.path.join(HERE, "results", "dashboard_data.json"), "w") as f:
        json.dump(out, f)
    won = sum(gm["nn_won"] for gm in games) / len(games)
    print("%d games in %.0f s; the network won %.1f%%" % (len(games), time.time() - t0, 100 * won))


if __name__ == "__main__":
    main()
