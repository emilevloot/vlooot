"""
Let the network play against itself and record every game, for the
dashboard (make_dashboard.py) and the replay in the game page
(index.html?replay=...).

    python selfplay_record.py --games 120 --model s2_r19

Per game:
  - the state after every turn (Game.save_state without the land and the
    log, which are stored once), so the game page can show any turn;
  - per turn: what the player did, the network's win chance and expected
    margin (for player 1, before the turn), and a deeper check of the move.

The deeper check (like a chess engine looking over a game): the player
chose the move its network liked best (one turn deep). Afterwards we take
the 3 best moves and let the opponent answer each with ITS best move; the
network judges where each line ends. When another move ends clearly better
than the one played, the turn is marked as a mistake:
    loss >= 3 points: mistake, >= 6 points: big mistake.

Written: selfplay_data.json (summary for the dashboard) and
replays/<seed>.json (one file per game, for the replay).
"""

import argparse
import copy
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = ["wood", "sheep", "gold", "axe", "house", "watchtower", "castle"]
CATS = ["castle", "watchtower", "house", "gold", "sheep", "wood"]
NL = {"wood": "hout", "sheep": "schaap", "gold": "goud", "axe": "bijl", "house": "huis",
      "watchtower": "wachttoren", "castle": "kasteel", "forest": "bos", "field": "weide",
      "mountain": "berg", "battlefield": "slagveld"}
MISTAKE, BIG = 3.0, 6.0
DEEP_TOP = 3


def counts(p):
    c = dict.fromkeys(ITEMS, 0)
    for it in p.fjord.values():
        if it["kind"] in ("res", "bld"):
            c[it["type"]] += 1
    return c


def ships(p):
    return {k: dict(it) for k, it in p.fjord.items() if it["kind"] == "ship"}


def snapshot(g):
    """g.save_state() shares lists and dicts with the running game: copy it,
    or earlier turns would change along with the game."""
    return json.loads(json.dumps(g.save_state()))


def key(c):
    return "%d,%d" % c


def describe(L, g, cand, me):
    """A move (a candidate of NNBot) in words, plus the facts behind it."""
    opt, tiles, opt2, tiles2, (slot, cell, tr), g4 = cand
    p0, p1 = g.players[me], g4.players[me]
    c0, c1 = counts(p0), counts(p1)
    got = {k: c1[k] - c0[k] for k in ITEMS if c1[k] != c0[k]}
    s0 = ships(p0)
    new = [it for k, it in ships(p1).items() if k not in s0]
    cells, parts, shields = [], [], []
    for o in (opt, opt2):
        if o is None:
            continue
        c, occ, dbl = o
        cells.append(key(c))
        parts.append(NL[g.land[c]] + (" (bezet, schild)" if occ else "") + (" (dubbel, schild)" if dbl else ""))
        if occ:
            shields.append("occupy")
        if dbl:
            shields.append("double")
    if opt2 is not None:
        shields.append("extra")
    text = " + ".join(parts) if parts else "geen Viking"
    if got:
        text += " → " + ", ".join("%d %s" % (n, NL[k]) for k, n in got.items())
    for it in new:
        text += "; schip %s +%d" % (NL[it["cat"]], it["bonus"])
    if tr is not None:
        text += "; trofee %d" % L.TROPHIES[tr][1]
    return {"text": text, "cells": cells, "got": got, "shields": shields,
            "ship": [{"cat": it["cat"], "bonus": it["bonus"]} for it in new],
            "trophy": L.TROPHIES[tr][1] if tr is not None else None}


def deep_check(bot, cands, vals, me):
    """Depth 2 over the best few moves: (index of the best line, value of
    every checked line {index: value})."""
    top = list(np.argsort(-vals)[:DEEP_TOP])
    after = []
    for k in top:
        g4 = cands[k][5]
        if g4.game_over or g4.current == me:
            after.append(g4)
            continue
        g5 = copy.deepcopy(g4)
        oc, ov = bot._candidates(g5)
        after.append(oc[int(np.argmax(ov))][5] if oc else g5)
    v2 = bot._judge(after, me)
    return {int(k): float(v) for k, v in zip(top, v2)}


def play(args):
    seed, model = args
    import looot as L
    import nn_bot
    import review
    g = L.Game([("Netwerk 1", True), ("Netwerk 2", True)], seed, L.read_layout_file())
    for p in g.players:
        p.nn = True
    bot = nn_bot.NNBot(random.Random(seed), model)
    net, enc = bot.net, bot.enc
    states = [snapshot(g)]
    alts = {}                  # turn -> the state after the better move
    turns = []
    t = 0
    while not g.game_over:
        me = g.current
        land, fjord, glob = enc.encode(g, 0)
        margin, win = net(land[None], fjord[None], glob[None])
        row = {"t": t, "who": me, "win0": round(float(win[0]), 4), "margin0": round(float(margin[0]), 2),
               "vikings_left": g.players[me].vikings_left}
        cands, vals = bot._candidates(g)
        if not cands:
            g.end_turn()
            continue
        k = int(np.argmax(vals))
        d = describe(L, g, cands[k], me)
        row.update(d)
        row["text_en"] = review.describe(g, cands[k][5], me, "en")["text"]   # for the game page
        row["value"] = round(float(vals[k]), 2)
        v2 = deep_check(bot, cands, vals, me)
        best = max(v2, key=v2.get)
        if k in v2:
            loss = v2[best] - v2[k]
            row["deep"] = round(v2[k], 2)
            if best != k and loss >= MISTAKE:
                alt = describe(L, g, cands[best], me)
                alt_en = review.describe(g, cands[best][5], me, "en")["text"]
                if alt_en == row["text_en"]:
                    alt_en += (" (another space)" if alt["cells"] != d["cells"]
                               else " (other spaces on the fjord)")
                if alt["text"] == d["text"]:
                    alt["text"] += (" (op een ander veld)" if alt["cells"] != d["cells"]
                                    else " (tegels/schip op andere plekken in de fjord)")
                alts[str(t)] = snapshot(cands[best][5])
                row["mistake"] = {"loss": round(loss, 2), "level": 2 if loss >= BIG else 1,
                                  "better": alt["text"], "better_en": alt_en,
                                  "better_cells": alt["cells"],
                                  "better_now": round(float(vals[best]), 2),
                                  "better_deep": round(v2[best], 2)}
        bot._apply(g, cands[k])
        states.append(snapshot(g))
        turns.append(row)
        t += 1
    final = []
    for p in g.players:
        sc = p.score()
        final.append({"total": sc["total"], "sites": sc["sites"], "trophy": sc["trophy"],
                      "penalty": sc["penalty"], **{c: sc["rows"][c]["points"] for c in CATS},
                      "filled": sum(1 for it in p.fjord.values() if it["kind"] == "ship" and it["filled"]),
                      "unfilled": sc["unfilled"]})
    w = g.winners()
    log = g.log
    replay = {"seed": seed, "model": model, "land": states[0]["land"], "log": log,
              "states": [], "turns": turns, "final": final, "winners": w}
    def small(s):
        s = dict(s)
        s.pop("land")
        s["log_n"] = len(s.pop("log"))
        return s
    replay["states"] = [small(s) for s in states]
    for s in alts.values():          # (their log went another way: keep its end)
        s.pop("land")
        s["log"] = s["log"][-14:]
    replay["alts"] = alts
    with open(os.path.join(HERE, "replays", "%d.json" % seed), "w") as f:
        json.dump(replay, f, separators=(",", ":"))
    return {"seed": seed, "turns": turns, "final": final, "winners": w}


def main():
    ap = argparse.ArgumentParser(description="Record self-play games for the dashboard and the replay.")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--model", default="s2_r19")
    ap.add_argument("--first-seed", type=int, default=41_000_000)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "replays"), exist_ok=True)
    for f in os.listdir(os.path.join(HERE, "replays")):
        if f.endswith(".json"):
            os.remove(os.path.join(HERE, "replays", f))
    t0 = time.time()
    jobs = [(a.first_seed + i, a.model) for i in range(a.games)]
    workers = max(1, min(a.games, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as ex:
        games = list(ex.map(play, jobs))
    out = {"model": a.model, "made": time.strftime("%Y-%m-%d %H:%M"), "games": games}
    with open(os.path.join(HERE, "selfplay_data.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    n_m = sum(1 for gm in games for r in gm["turns"] if "mistake" in r)
    print("%d games in %.0f s; %d mistakes found (%.1f per game)"
          % (len(games), time.time() - t0, n_m, n_m / len(games)))


if __name__ == "__main__":
    main()
