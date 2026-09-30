"""
The first 4-player training, starting from the best 2-player network.

    python tools/run_4p.py --hours 2

1. The best 2-player network: of the networks in the head-to-head matches
   of compare_runs.py (compare_*.json), the one with the best results
   against the others (or --from <model> to choose yourself).
2. It becomes a 4-player network of the same kind (transfer_mp.py): it
   keeps everything it learned.
3. The curriculum's last stage (the full game) with 4 players, starting from
   that network: greedy games, then self-play rounds with the best network
   moving on, until the time is used up; then the final tests against 3
   greedy players (a random player would win 25%).
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import glob
import json
import subprocess

HERE = ROOT


def best_from_matches():
    """The network that did best in the head-to-head matches: for every
    network the average of its win rates against the others."""
    scores = {}
    for f in glob.glob(os.path.join(HERE, "results", "compare_*.json")):
        m = json.load(open(f))
        for me, other in (("a", "b"), ("b", "a")):
            scores.setdefault(m[me], []).append(m[me + "_win"])
    if not scores:
        raise SystemExit("No compare_*.json yet: run compare_runs.py first, or pass --from.")
    avg = {k: sum(v) / len(v) for k, v in scores.items()}
    for k in sorted(avg, key=avg.get, reverse=True):
        print("  %-16s %.1f%% on average in its matches" % (k, 100 * avg[k]))
    return max(avg, key=avg.get)


def run(args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser(description="4-player training from the best 2-player network.")
    ap.add_argument("--from", dest="src", default=None)
    ap.add_argument("--prefix", default="m4")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--players", type=int, default=4)
    a = ap.parse_args()
    src = a.src or best_from_matches()
    print("Starting from", src, flush=True)
    start = a.prefix + "_start"
    if not os.path.exists(os.path.join(HERE, "models", start + ".pt")):
        stats = a.prefix + "_scale"
        if not glob.glob(os.path.join(HERE, "data", stats + "_*.npz")):
            run(["gen_data.py", "--players", str(a.players), "--games", "400", "--name", stats,
                 "--seed", "91000000", "--batch", "400"])
        run(["tools/transfer_mp.py", "--from", src, "--data", stats, "--name", start])
    # 4-player games are about 4x the positions of 2-player games: fewer
    # games per round, so the training data still fits in memory
    run(["curriculum.py", "--prefix", a.prefix, "--players", str(a.players), "--stages", "full",
         "--init-model", start, "--hours", str(a.hours),
         "--greedy-games", "1500", "--selfplay-games", "2000", "--window", "2",
         "--batch", "2048", "--arena-games", "300", "--arena-games-long", "400"])


if __name__ == "__main__":
    main()
