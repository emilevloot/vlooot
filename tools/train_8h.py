"""
An 8-hour training session, one step after the other:

  1. 2 players (about 4.5 hours): self-play with a champion, continuing the
     s2 run from its champion (s2_r19). Changes from before: the networks
     overfitted after the first epoch, so a lower learning rate and 2 epochs;
     positions the champion judged far off count more (--surprise); the
     match that decides a new champion has 600 games (less luck).
  2. 3 and 4 players (about 3 hours): the t4 run, continuing from t4_start,
     now with 3000 games a round and the last 4 rounds of games (it had only
     3000 games in all to learn from, which was too little).
  3. The dashboard: new self-play games of the 2-player champion.

    python tools/train_8h.py      (or start/start_8h.bat)

Progress: results/curriculum_s2_log.txt, results/curriculum_t4_log.txt and
results/train_8h_log.txt.
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
import subprocess
import time

HERE = ROOT
LOG = os.path.join(HERE, "results", "train_8h_log.txt")


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args):
    log("> " + " ".join(args))
    t = time.time()
    code = subprocess.run([sys.executable] + args, cwd=HERE).returncode
    log("  finished in %.0f min (exit code %d)" % ((time.time() - t) / 60, code))
    return code


def champion(prefix, fallback):
    try:
        res = json.load(open(os.path.join(HERE, "results", "curriculum_%s_results.json" % prefix)))
        return res["stages"]["full"].get("champion") or fallback
    except (OSError, KeyError, ValueError):
        return fallback


def main():
    t0 = time.time()
    log("=== 8-hour session ===")
    run(["selfplay_train.py", "--prefix", "s2", "--init", "s2_r19", "--hours", "4.5",
         "--games", "8000", "--window", "4", "--epochs", "2", "--lr", "5e-4",
         "--surprise", "2", "--match-games", "600"])
    best2 = champion("s2", "s2_r19")
    log("2-player champion: %s" % best2)
    left = 8.0 - (time.time() - t0) / 3600
    hours = max(1.0, left - 0.4)
    run(["selfplay_train.py", "--prefix", "t4", "--init", "t4_start", "--players", "3,4",
         "--games", "3000", "--window", "4", "--epochs", "2", "--lr", "3e-4",
         "--surprise", "2", "--batch", "2048", "--match-games", "400",
         "--arena-games", "200", "--hours", "%.2f" % hours])
    log("3-4-player champion: %s" % champion("t4", "t4_start"))
    run(["tools/selfplay_record.py", "--games", "100", "--model", best2])
    run(["tools/make_dashboard.py"])
    log("=== done in %.1f hours ===" % ((time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
