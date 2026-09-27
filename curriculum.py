"""
Teach the value network the game step by step ("curriculum learning").

    python curriculum.py --prefix c2          # all stages; continues where it stopped
    python curriculum.py --prefix c2 --report # only (re)make the report
    python curriculum.py --prefix c3 --hours 6 --rounds longships=12
                         # a time budget: the last stage keeps playing
                         # self-play rounds until the time is almost up

The stages are looot.RULE_STAGES: resources -> buildings -> sites ->
longships -> full. For every stage:

  1. data      the greedy player plays games under the stage's rules
  2. train     the network learns from them, starting from the best network
               of the previous stage (fine-tuning): what it learned before
               is kept, only the new rule has to be learned
  3. rounds    self-play: the newest network plays itself, then a new network
               is trained on the greedy data plus the latest self-play data.
               With --td, the answer key is a mix of the final result and
               the previous network's judgement a few turns later.
  4. arena     after every step: the network against the greedy player,
               under the stage's rules. The best network moves on.

Results: curriculum_<prefix>_results.json and curriculum_<prefix>_log.txt,
and curriculum_report.html (graphs, made by curriculum_report.py).
The first run (encoding version 1, no TD) is kept as prefix "cur".
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

import looot as L          # noqa: E402

STAGES = [s for s, _ in L.RULE_STAGES]
ROUNDS = {"resources": 0, "buildings": 1, "sites": 1, "longships": 6, "full": 8}
FINALS_RESERVE = 15 * 60        # seconds kept free for the final tests


class Run:
    def __init__(self, prefix):
        self.prefix = prefix
        self.log_file = os.path.join(HERE, "curriculum_%s_log.txt" % prefix)
        self.res_file = os.path.join(HERE, "curriculum_%s_results.json" % prefix)
        self.results = (json.load(open(self.res_file)) if os.path.exists(self.res_file)
                        else {"prefix": prefix, "stages": {}, "final": {}})

    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save(self):
        with open(self.res_file, "w") as f:
            json.dump(self.results, f, indent=1)

    def run(self, args):
        """Run one of our scripts; stop everything if it fails."""
        t = time.time()
        p = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True)
        out = (p.stdout + p.stderr).strip().splitlines()
        if p.returncode != 0:
            self.log("FAILED: %s\n%s" % (" ".join(args), "\n".join(out[-25:])))
            raise SystemExit(1)
        return out, time.time() - t


def keep_awake():
    """Ask Windows not to go to sleep while this program runs (the way a video
    player does). No setting is changed: it ends when the program ends."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def kept(out):
    """The 'Keeping epoch N (x points off)' line of train_nn.py, as a number."""
    for l in out:
        if l.startswith("Keeping epoch"):
            return float(l.split("(")[1].split()[0])
    return None


def arena_vs_greedy(model, stage, games, seed):
    import arena
    res = arena.run(["nn:" + model, "greedy"], games, seed,
                    L.read_layout_file(), rules=stage)
    rows = [r for g in res for r in g if r["name"] == "nn:" + model]
    n = len(rows)
    out = {"games": games}
    for k in ("win", "margin", "score"):
        out[k] = sum(r[k] for r in rows) / n
    out["win_ci"] = 1.96 * (out["win"] * (1 - out["win"]) / n) ** 0.5
    out["penalty"] = sum(r["parts"]["penalty"] for r in rows) / n
    out["ms_per_turn"] = 1000 * sum(r["think"] for r in rows) / max(1, sum(r["turns"] for r in rows))
    return out


def stage_run(R, si, a, prev):
    st = STAGES[si]
    P = R.prefix
    seed = 1_000_000 * (si + 1) + (0 if P == "cur" else 50_000_000)
    games_arena = a.arena_games if a.rounds[st] < 3 else a.arena_games_long
    r = R.results["stages"].setdefault(st, {"steps": []})
    R.log("=== stage %d/%d: %s ===" % (si + 1, len(STAGES), st))

    def record(model, kind, mae, extra=None):
        res = arena_vs_greedy(model, st, games_arena, seed + 900_000)
        step = {"model": model, "kind": kind, "test_mae": mae, "arena": res}
        if extra:
            step.update(extra)
        r["steps"].append(step)
        R.save()
        R.log("  %-22s wins %5.1f%% ±%.1f, margin %+6.1f, score %5.1f, longship penalty "
              "%5.1f, %4.0f ms/turn (test error %s)"
              % (model, 100 * res["win"], 100 * res["win_ci"], res["margin"], res["score"],
                 res["penalty"], res["ms_per_turn"], "%.2f" % mae if mae else "-"))

    # 1. greedy data
    gdata = "%s_%s_g" % (P, st)
    _, t = R.run(["gen_data.py", "--games", str(a.greedy_games), "--rules", st,
                  "--name", gdata, "--seed", str(seed), "--batch", "2000"])
    R.log("  data: %d greedy games in %.0f s" % (a.greedy_games, t))

    # 2. first network of this stage (no TD: the previous stage's network
    #    doesn't know this stage's new rule, so its judgement can't be used)
    model = "%s_%s_a" % (P, st)
    args = ["train_nn.py", "--data", gdata, "--name", model, "--epochs", str(a.epochs)]
    args += ["--init", prev, "--lr", "2e-3"] if prev else ["--stats-from", P + "_stats"]
    out, t = R.run(args)
    R.log("  trained %s in %.0f s" % (model, t))
    record(model, "greedy data", kept(out))

    # 3. self-play rounds
    sp = []
    last = st == STAGES[-1]
    n_rounds = a.max_rounds if (last and a.hours) else min(a.rounds[st], a.max_rounds)
    round_time = 0
    for k in range(1, n_rounds + 1):
        if a.hours:
            # stop when the next round would eat into the time for the finals
            left = a.deadline - time.time()
            if left < round_time + FINALS_RESERVE:
                R.log("  time budget reached: no more rounds (%.0f min left)" % (left / 60))
                break
        t_round = time.time()
        name = "%s_%s_sp%d" % (P, st, k)
        _, t = R.run(["gen_data.py", "--games", str(a.selfplay_games), "--rules", st,
                      "--player", "nn:" + model, "--name", name,
                      "--seed", str(seed + 100_000 * k), "--batch", "750"])
        sp.append(name)
        new = "%s_%s_r%d" % (P, st, k)
        data = [gdata] + sp[-a.window:]
        args = ["train_nn.py", "--data"] + data + ["--name", new, "--epochs", str(a.epochs),
                                                   "--init", model, "--lr", "2e-3"]
        if a.td:
            args += ["--td", str(a.td), "--td-mix", str(a.td_mix)]
        out, t2 = R.run(args)
        R.log("  round %d: %d self-play games in %.0f s, trained %s in %.0f s"
              % (k, a.selfplay_games, t, new, t2))
        record(new, "self-play round %d" % k, kept(out))
        model = new
        round_time = time.time() - t_round

    # the best network of the stage moves on
    best = max(r["steps"], key=lambda s: s["arena"]["win"] + 0.002 * s["arena"]["margin"])
    r["best"] = best["model"]
    R.save()
    R.log("  best of stage %s: %s (%.1f%% wins)" % (st, best["model"], 100 * best["arena"]["win"]))
    return best["model"]


def finals(R, a):
    """Extra tests of the best full-game network."""
    best = R.results["stages"]["full"]["best"]
    tests = [("%s vs greedy" % best, ["nn:" + best, "greedy"], a.arena_games_long),
             ("%s+2 (looks 2 turns ahead) vs greedy" % best, ["nn:%s+2" % best, "greedy"],
              min(200, a.arena_games_long)),
             ("%s vs original greedy" % best, ["nn:" + best, "original"], a.arena_games_long)]
    import arena
    for label, lineup, games in tests:
        res = arena.run(lineup, games, 88_000_000, L.read_layout_file())
        rows = [r for g in res for r in g if r["name"] == lineup[0]]
        n = len(rows)
        out = {"games": games, "win": sum(r["win"] for r in rows) / n,
               "margin": sum(r["margin"] for r in rows) / n,
               "score": sum(r["score"] for r in rows) / n,
               "penalty": sum(r["parts"]["penalty"] for r in rows) / n,
               "ms_per_turn": 1000 * sum(r["think"] for r in rows) / max(1, sum(r["turns"] for r in rows))}
        out["win_ci"] = 1.96 * (out["win"] * (1 - out["win"]) / n) ** 0.5
        R.results["final"][label] = out
        R.save()
        R.log("FINAL %s: wins %.1f%% ±%.1f, margin %+.1f, longship penalty %.1f, %.0f ms/turn"
              % (label, 100 * out["win"], 100 * out["win_ci"], out["margin"], out["penalty"],
                 out["ms_per_turn"]))


def main():
    ap = argparse.ArgumentParser(description="Curriculum training.")
    ap.add_argument("--prefix", default="c2", help="name for this run's data and models")
    ap.add_argument("--greedy-games", type=int, default=8000)
    ap.add_argument("--selfplay-games", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--window", type=int, default=4,
                    help="train on the greedy data + the last N self-play sets")
    ap.add_argument("--td", type=int, default=2,
                    help="TD: look N turns ahead for the target (0 = off)")
    ap.add_argument("--td-mix", type=float, default=0.5)
    ap.add_argument("--arena-games", type=int, default=200)
    ap.add_argument("--arena-games-long", type=int, default=300,
                    help="arena games in stages with many rounds (more precise)")
    ap.add_argument("--report", action="store_true", help="only make the report")
    ap.add_argument("--max-rounds", type=int, default=99, help="cap on rounds per stage")
    ap.add_argument("--rounds", default="",
                    help="self-play rounds per stage, e.g. buildings=3,longships=12 "
                         "(the rest as in ROUNDS)")
    ap.add_argument("--hours", type=float, default=0,
                    help="time budget: the last stage keeps going until it is used up")
    a = ap.parse_args()
    rounds = dict(ROUNDS)
    for part in filter(None, a.rounds.split(",")):
        st, n = part.split("=")
        if st not in rounds:
            raise SystemExit("unknown stage %r (stages: %s)" % (st, ", ".join(STAGES)))
        rounds[st] = int(n)
    a.rounds = rounds
    a.deadline = time.time() + 3600 * a.hours

    R = Run(a.prefix)
    if not a.report:
        keep_awake()
        # input scaling for every stage comes from full-game data
        if not os.path.exists(os.path.join(HERE, "data", a.prefix + "_stats_000.npz")):
            R.run(["gen_data.py", "--games", str(min(2000, a.greedy_games)),
                   "--name", a.prefix + "_stats", "--seed", "77000000",
                   "--batch", str(min(2000, a.greedy_games))])
        prev = None
        for si, st in enumerate(STAGES):
            done = R.results["stages"].get(st, {}).get("best")
            if done:
                prev = done
                continue
            R.results["stages"].pop(st, None)
            prev = stage_run(R, si, a, prev)
        if not R.results["final"]:
            finals(R, a)
        R.log("Curriculum finished.")
    import curriculum_report
    print("Report:", curriculum_report.make())


if __name__ == "__main__":
    main()
