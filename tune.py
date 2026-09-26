"""
Tune the computer player's numbers (BOT_WEIGHTS in looot.py) by playing
lots of games in the arena.

    python tune.py                      # tune for 30 minutes
    python tune.py --minutes 120        # longer
    python tune.py --apply              # write the best weights into looot.py

How it works, every round ("generation"):
  1. Make a few variations of the current best weights (the champion),
     each with some numbers nudged up or down.
  2. Each variation plays 3-player games against two champions, all on the
     same boards and longship draws. It gets a score: how many points it
     makes more than the champions in the same games.
  3. The best variation plays a second, bigger set of new games. Only if
     it is still clearly ahead there does it become the new champion.

Progress is saved in tune_best.json (so you can stop with Ctrl+C and run
again to continue) and logged in tune_log.txt. Every round is also stored
in tune_history.json, and at the end tune_report.html is made with graphs
of the scores and of every weight over the rounds.
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

import arena
import looot as L

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_FILE = os.path.join(HERE, "tune_best.json")
LOG_FILE = os.path.join(HERE, "tune_log.txt")
HISTORY_FILE = os.path.join(HERE, "tune_history.json")

# name: (lowest, highest, whole numbers?)
RANGES = {
    "ship_base": (0.3, 1.0, False),
    "ship_per_missing": (0.0, 0.6, False),
    "ship_late": (0.0, 0.8, False),
    "ship_min": (0.0, 0.6, False),
    "ship_hopeless": (0.3, 4.0, False),
    "ship_hopeless_chance": (0.0, 0.4, False),
    "ship_future_res": (0.0, 1.5, False),
    "ship_future_bld": (0.0, 1.0, False),
    "site_power": (0.5, 4.0, False),
    "site_early": (0.0, 2.0, False),
    "site_late": (0.0, 2.0, False),
    "site_late_left": (0, 6, True),
    "trophy_ready": (0.2, 1.2, False),
    "trophy_progress": (0.0, 1.2, False),
    "shield_early": (0.0, 6.0, False),
    "shield_late": (0.0, 4.0, False),
    "shield_late_left": (0, 8, True),
    "chain": (0.0, 4.0, False),
    "tile_need": (1.0, 30.0, False),
    "tile_block": (0.0, 15.0, False),
    "tile_open": (0.0, 2.0, False),
    "shipcell_match": (0.5, 20.0, False),
    "shipcell_room": (0.0, 5.0, False),
    "extra_gain": (-2.0, 10.0, False),
    "extra_force_left": (0, 8, True),
    "take_ship_margin": (-5.0, 8.0, False),
    "trophy_wait_left": (0, 8, True),
    "noise": (0.0, 2.0, False),
}
assert set(RANGES) == set(L.BOT_WEIGHTS), "RANGES must list every weight"


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def mutate(w, rng, sigma, share=0.35):
    """A copy of w with about `share` of the numbers nudged."""
    out = dict(w)
    keys = [k for k in RANGES if rng.random() < share] or [rng.choice(list(RANGES))]
    for k in keys:
        lo, hi, whole = RANGES[k]
        if whole:
            out[k] = int(min(hi, max(lo, out[k] + rng.choice([-2, -1, 1, 2]))))
        else:
            v = out[k] + rng.gauss(0, sigma * (hi - lo))
            out[k] = round(min(hi, max(lo, v)), 3)
    return out


def advantage(results, name):
    """Per game: points of `name` minus the average of the others."""
    adv = []
    for game in results:
        mine = [r["score"] for r in game if r["name"] == name]
        other = [r["score"] for r in game if r["name"] != name]
        adv.append(statistics.mean(mine) - statistics.mean(other))
    return adv


def win_share(results, name):
    return statistics.mean(r["win"] for g in results for r in g if r["name"] == name)


def avg_score(results, name):
    return statistics.mean(r["score"] for g in results for r in g if r["name"] == name)


def save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=1)


def mean_ci(v):
    m = statistics.mean(v)
    return m, 1.96 * statistics.stdev(v) / math.sqrt(len(v))


def match(cand, champ, games, seed, layout, pool):
    lineup = [("cand", cand), ("champ", champ), ("champ", champ)]
    return arena.run(lineup, games, seed, layout, pool=pool)


def apply_to_looot(weights):
    """Write `weights` into the BOT_WEIGHTS block of looot.py, keeping the
    comments."""
    path = os.path.join(HERE, "looot.py")
    src = open(path, encoding="utf-8").read()
    start = src.index("BOT_WEIGHTS = {")
    end = src.index("\n}\n", start)
    block = src[start:end]
    for k, v in weights.items():
        val = str(int(v)) if RANGES[k][2] else repr(float(v))
        block, n = re.subn(r'("%s":\s*)[-0-9.e]+' % re.escape(k),
                           lambda m: m.group(1) + val, block)
        assert n == 1, k
    with open(path, "w", encoding="utf-8") as f:
        f.write(src[:start] + block + src[end:])


def main():
    ap = argparse.ArgumentParser(description="Tune BOT_WEIGHTS by self-play.")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--candidates", type=int, default=8,
                    help="variations tried per round (default 8)")
    ap.add_argument("--quick-games", type=int, default=90,
                    help="games per variation (default 90)")
    ap.add_argument("--check-games", type=int, default=300,
                    help="games to confirm a new champion (default 300)")
    ap.add_argument("--sigma", type=float, default=0.12,
                    help="size of the nudges, as a share of each range")
    ap.add_argument("--random-boards", action="store_true",
                    help="tune on random boards instead of boards.json")
    ap.add_argument("--apply", action="store_true",
                    help="only write tune_best.json into looot.py and stop")
    a = ap.parse_args()

    if a.apply:
        best = json.load(open(BEST_FILE, encoding="utf-8"))["weights"]
        apply_to_looot(best)
        print("Wrote the tuned weights into BOT_WEIGHTS in looot.py.")
        return

    layout = None if a.random_boards else L.read_layout_file()
    state = {"weights": dict(L.BOT_WEIGHTS), "generation": 0, "seed": 1000}
    if os.path.exists(BEST_FILE):
        state = json.load(open(BEST_FILE, encoding="utf-8"))
        log("Continuing from tune_best.json (round %d)." % state["generation"])
    champ = state["weights"]
    rng = random.Random(state["seed"])
    sigma = a.sigma
    stop_at = time.time() + a.minutes * 60
    workers = max(1, (os.cpu_count() or 2) - 1)
    hist = {"start_weights": dict(champ), "ranges": RANGES, "rounds": []}
    if os.path.exists(HISTORY_FILE) and state["generation"] > 0:
        hist = json.load(open(HISTORY_FILE, encoding="utf-8"))
    t0 = time.time() - (hist["rounds"][-1]["time_s"] if hist["rounds"] else 0)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while time.time() < stop_at:
            state["generation"] += 1
            gen = state["generation"]
            seed = state["seed"] = state["seed"] + 1000
            cands = [mutate(champ, rng, sigma) for _ in range(a.candidates)]
            scored = []
            champ_scores = []
            for c in cands:
                res = match(c, champ, a.quick_games, seed, layout, pool)
                m, ci = mean_ci(advantage(res, "cand"))
                scored.append((m, ci, win_share(res, "cand"), c,
                               avg_score(res, "cand")))
                champ_scores.append(avg_score(res, "champ"))
            rnd = {"round": gen, "sigma": sigma,
                   "champ_score": statistics.mean(champ_scores),
                   "variations": [{"adv": s[0], "ci": s[1], "win": s[2],
                                   "score": s[4]} for s in scored],
                   "check": None, "accepted": False}
            scored.sort(key=lambda t: -t[0])
            m, ci, ws, best, _ = scored[0]
            log("round %d: best variation %+.1f pts (±%.1f), wins %.0f%%; "
                "others %s" % (gen, m, ci, 100 * ws,
                               " ".join("%+.1f" % s[0] for s in scored[1:])))
            if m <= 0:
                sigma = max(0.03, sigma * 0.9)
            else:
                res = match(best, champ, a.check_games, seed + 500, layout, pool)
                m2, ci2 = mean_ci(advantage(res, "cand"))
                ws2 = win_share(res, "cand")
                rnd["check"] = {"adv": m2, "ci": ci2, "win": ws2,
                                "score": avg_score(res, "cand"),
                                "champ_score": avg_score(res, "champ")}
                if m2 - ci2 > 0:
                    changed = {k: (champ[k], best[k]) for k in best if best[k] != champ[k]}
                    champ = best
                    state["weights"] = champ
                    rnd["accepted"] = True
                    log("  NEW CHAMPION: %+.1f pts (±%.1f), wins %.0f%% (fair 33%%). "
                        "Changed: %s" % (m2, ci2, 100 * ws2, ", ".join(
                            "%s %s->%s" % (k, o, n) for k, (o, n) in changed.items())))
                else:
                    log("  not confirmed: %+.1f pts (±%.1f), wins %.0f%%" %
                        (m2, ci2, 100 * ws2))
                    sigma = max(0.03, sigma * 0.95)
            rnd["weights"] = dict(champ)
            rnd["time_s"] = time.time() - t0
            hist["rounds"].append(rnd)
            save_history(hist)
            with open(BEST_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=1)

        # Final check against the original hand-picked weights
        res = arena.run([("tuned", champ), ("original", arena.ORIGINAL_WEIGHTS),
                         ("original", arena.ORIGINAL_WEIGHTS)],
                        600, 777777, layout, pool=pool)
        m, ci = mean_ci(advantage(res, "tuned"))
        hist["final"] = {"games": 600, "adv": m, "ci": ci,
                         "win": win_share(res, "tuned"),
                         "score": avg_score(res, "tuned"),
                         "original_score": avg_score(res, "original")}
        save_history(hist)
        log("FINAL vs original (600 games): tuned wins %.1f%% (fair 33.3%%), "
            "%+.1f pts per game (±%.1f)" % (100 * win_share(res, "tuned"), m, ci))
    import tune_report
    tune_report.make()
    log("Graphs: tune_report.html. Best weights are in tune_best.json. "
        "Run  python tune.py --apply  to put them into looot.py.")


if __name__ == "__main__":
    main()
