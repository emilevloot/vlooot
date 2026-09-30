"""
Let the best full-game networks of two curriculum runs play each other.

    python compare_runs.py c5v c5t --games 500

Both runs' best network (curriculum_<prefix>_results.json) play the same
boards, each seat order equally often. Prints the result and saves it to
compare_<a>_<b>.json; the report (curriculum_report.html) is made again.
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import json
import time

import arena
import looot as L

HERE = ROOT


def best_of(prefix):
    res = json.load(open(os.path.join(HERE, "results", "curriculum_%s_results.json" % prefix)))
    return res["stages"]["full"]["best"]


def main():
    ap = argparse.ArgumentParser(description="Head-to-head of two curriculum runs.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=500)
    a = ap.parse_args()
    ma, mb = best_of(a.a), best_of(a.b)
    lineup = ["nn:" + ma, "nn:" + mb]
    t = time.time()
    res = arena.run(lineup, a.games, 99_000_000, L.read_layout_file())
    out = {"games": a.games, "a": ma, "b": mb}
    for name, key in ((lineup[0], "a"), (lineup[1], "b")):
        rows = [r for g in res for r in g if r["name"] == name]
        n = len(rows)
        win = sum(r["win"] for r in rows) / n
        out[key + "_win"] = win
        out[key + "_win_ci"] = 1.96 * (win * (1 - win) / n) ** 0.5
        out[key + "_margin"] = sum(r["margin"] for r in rows) / n
        out[key + "_score"] = sum(r["score"] for r in rows) / n
    with open(os.path.join(HERE, "results", "compare_%s_%s.json" % (a.a, a.b)), "w") as f:
        json.dump(out, f, indent=1)
    line = ("%s  %s (%s) vs %s (%s): %.1f%% ±%.1f wins for %s, margin %+.1f, "
            "scores %.1f - %.1f (%d games, %.0f s)"
            % (time.strftime("%H:%M:%S"), ma, a.a, mb, a.b, 100 * out["a_win"],
               100 * out["a_win_ci"], ma, out["a_margin"], out["a_score"], out["b_score"],
               a.games, time.time() - t))
    print(line)
    with open(os.path.join(HERE, "results", "compare_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    import curriculum_report
    print("Report:", curriculum_report.make())


if __name__ == "__main__":
    main()
