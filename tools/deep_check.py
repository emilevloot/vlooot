"""
How far is the network's quick estimate from the deep analysis?

While a game is played the page shows every player's chance to win: the
network asked once, from each seat (review.shares: the bar and the graph;
the coach's bubble shows the raw chance of the player who moved), and the
coach judges each move by the network's estimate of the final margin. The
deep analysis (analysis.py) plays the rest of the game out many times
instead, with the network on every seat. This measures how far apart the
two are, on the positions of real games:

  positions  games of the network against itself and against the greedy
             player (2 players: s2_r32; 3-4 players: t4_r8). After every
             turn: the network's estimate of each player's chance to win
             and final margin, and N playouts from there.
  moves      the greedy player's moves (2 players): the points the move
             lost by the coach's estimate, against the deep analysis (the
             move played and the best 3 others, played out in pairs). In
             a player's last 2 turns also the exact count (endgame.py).

The playouts have their own luck: with N of them a chance to win is known
to a few percent. The typical difference ("rmse") is corrected for that:
the mean square difference minus the playouts' own variance.

    python tools/deep_check.py                 (about 25 minutes)
    python tools/deep_check.py --games2 2 --games4 1 --n2 8 --n4 4   (a quick try)

Each playout process may take a fifth of the GPU's memory (GPU_SHARE; 128
games at once need about 2 GB): 4 processes. With more than fit, Windows
moves GPU memory to the normal memory and everything crawls.

Writes results/deep_check.json and prints a summary.
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

MODELS = {2: "s2_r32", 3: "t4_r8", 4: "t4_r8"}
PHASES = [("start", 0.0, 0.2), ("early", 0.2, 0.4), ("middle", 0.4, 0.6),
          ("late", 0.6, 0.8), ("end", 0.8, 1.01)]
CLASSES = [("best", 0.5), ("excellent", 1.2), ("good", 2.0),
           ("inaccuracy", 4.0), ("mistake", 8.0), ("blunder", float("inf"))]
CHUNK = 12                  # starts per playout job (all of one game: the same futures)
GPU_SHARE = 0.2             # of the GPU's memory, at most, per playout process


def student(model, seed):
    """A decent player that slips: the network's choice, but in a third of
    its turns one of the network's next 7 best moves instead - close games
    with mistakes of every size, more like a person than the greedy player."""
    import nn_bot

    class Student(nn_bot.NNBot):
        def play_turn(self, g):
            cands, vals = self._candidates(g)
            if not cands:
                if g.phase == "actions":
                    g.end_turn()
                return
            order = np.argsort(-vals)
            k = order[0]
            if len(order) > 1 and self.rng.random() < 0.35:
                k = order[self.rng.randrange(1, min(8, len(order)))]
            self._apply(g, cands[int(k)])

    return Student(random.Random(seed), model)


def play_game(job):
    """One game: after every turn the network's estimate for every player;
    for the moves of the greedy players and the students (judge) the
    coach's estimate of the points lost and the best 3 other moves. In a
    worker process."""
    import looot as L
    import nn_bot
    import review
    import analysis
    n_pl, seats, seed, judge = job
    model = MODELS[n_pl]
    net = nn_bot.load_net(model)
    g = L.Game([("p%d" % i, True) for i in range(n_pl)], seed, L.read_layout_file())
    bots = [nn_bot.NNBot(random.Random(seed + i), model) if kind == "nn" else
            student(model, seed + i) if kind == "student" else L.Bot(random.Random(seed + i))
            for i, kind in enumerate(seats)]
    start_left = sum(p.vikings_left for p in g.players)
    positions, moves, mover = [], [], None
    while True:
        est = [review.evaluate(net, [g], s) for s in range(n_pl)]
        positions.append({"state": review.snapshot(g), "mover": mover,
                          "progress": 1.0 - sum(p.vikings_left for p in g.players) / start_left,
                          "margin": [float(e[0][0]) for e in est], "win": [float(e[1][0]) for e in est],
                          "shares": review.shares(net, g)})
        if g.game_over:
            break
        me = g.current
        g0 = L.Game.load_state(positions[-1]["state"])
        bots[me].play_turn(g)
        mover = me
        if not (judge and seats[me] != "nn"):
            continue
        g1 = L.Game.load_state(review.snapshot(g))
        games, vals = analysis.candidates(model, g0)
        if not games:
            continue
        pm = float(review.evaluate(net, [g1], me)[0][0])
        alts = [(a, v) for a, _, v in analysis.alternatives(g0, g1, games, vals, 3)]
        top_played = analysis.same_move(games[0], g1, me)
        if not top_played and not any(a is games[0] for a, _ in alts):
            alts = [(games[0], float(vals[0]))] + alts[:2]    # the best, in other fjord spaces
        mv = {"pos": len(positions) - 1, "me": me, "left": g0.players[me].vikings_left, "who": seats[me],
              "loss": max(0.0, float(vals[0]) - pm), "top_played": top_played,
              "alts": [review.snapshot(a) for a, _ in alts], "alt_vals": [v - pm for _, v in alts]}
        if n_pl == 2 and 1 <= g0.players[me].vikings_left <= 2:
            import endgame
            c = endgame.check(g0, g1, me, model, endgame.WIDE)
            if c is not None:
                mv["exact"] = float(c["loss"])
        moves.append(mv)
    return {"players": n_pl, "seats": seats, "seed": seed, "positions": positions, "moves": moves}


def play_out_job(job):
    """Play starts of one game out n times, all with the same futures (so
    they can be compared in pairs). In a worker process."""
    import looot as L
    import analysis
    key, n_pl, seed, n, states, parallel = job
    import torch
    if torch.cuda.is_available():
        # PyTorch keeps freed GPU memory for later; in several processes that
        # grows until the GPU is full (and Windows swaps it: everything crawls)
        torch.cuda.set_per_process_memory_fraction(GPU_SHARE)
    res = analysis.play_out([L.Game.load_state(s) for s in states], None, MODELS[n_pl], n, seed=seed,
                            parallel=parallel)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return key, [r.astype(np.float32) for r in res]


def classify(loss):
    return next(name for name, top in CLASSES if loss <= top)


def err(est, deep, var):
    """How far the estimates are from the playouts: mean absolute and
    mean signed difference, and the typical difference (root mean square)
    without the playouts' own luck."""
    d = np.asarray(est) - np.asarray(deep)
    noise = float(np.mean(var))
    return {"n": int(len(d)), "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d)),
            "rmse": float(np.sqrt(max(0.0, np.mean(d ** 2) - noise))), "noise": float(np.sqrt(noise)),
            "rmse_raw": float(np.sqrt(np.mean(d ** 2)))}


def summarize(games, deep):
    """The numbers: per group of player counts, how far the network's
    chances and margins are from the playouts, by phase of the game; how
    well its chances are calibrated; and how the coach's points lost
    compare with the deep analysis."""
    rows = []
    for gi, gm in enumerate(games):
        P = gm["players"]
        final = gm["positions"][-1]
        for i, pos in enumerate(gm["positions"][:-1]):
            r = deep[(gi, "pos", i)]
            n = len(r)
            m, w = r[:, :, 0], r[:, :, 1]
            for s in range(P):
                rows.append({"group": "2" if P == 2 else "3-4", "players": P, "progress": pos["progress"],
                             "mover": s == pos["mover"], "share": pos["shares"][s], "win": pos["win"][s],
                             "margin": pos["margin"][s], "d_win": float(w[:, s].mean()),
                             "d_win_var": float(w[:, s].var(ddof=1) / n), "d_margin": float(m[:, s].mean()),
                             "d_margin_var": float(m[:, s].var(ddof=1) / n), "spread": float(m[:, s].std(ddof=1)),
                             "final": final["margin"][s]})
    out = {}
    for group in ("2", "3-4"):
        R = [r for r in rows if r["group"] == group]
        if not R:
            continue
        A = {k: np.array([r[k] for r in R]) for k in R[0]}
        mo = A["mover"]
        res = {"positions": int(mo.sum()), "players": sorted({int(p) for p in A["players"]})}
        # the bar: every player's share (adds up to 1), against the playouts' share of wins
        res["bar"] = err(A["share"], A["d_win"], A["d_win_var"])
        # the bubble: the raw chance of the player who just moved
        res["bubble"] = err(A["win"][mo], A["d_win"][mo], A["d_win_var"][mo])
        # the margin of the player who just moved, against the playouts and against the real end
        res["margin"] = err(A["margin"][mo], A["d_margin"][mo], A["d_margin_var"][mo])
        res["margin_vs_end"] = float(np.sqrt(np.mean((A["margin"][mo] - A["final"][mo]) ** 2)))
        res["spread"] = float(np.sqrt(np.mean(A["spread"][mo] ** 2)))
        # big leads: above 1 the network squeezes the margin towards 0
        res["margin_slope"] = float(np.polyfit(A["margin"][mo], A["d_margin"][mo], 1)[0])
        close = mo & (np.abs(A["margin"]) < 15)
        res["margin_close"] = err(A["margin"][close], A["d_margin"][close], A["d_margin_var"][close])
        res["margin_far"] = err(A["margin"][mo & ~close], A["d_margin"][mo & ~close], A["d_margin_var"][mo & ~close])
        # the positions themselves, for charts: progress, the network's and the playouts' margin and chance
        res["points"] = [[round(float(A["progress"][i]), 3), round(float(A["margin"][i]), 1),
                          round(float(A["d_margin"][i]), 1), round(float(A["win"][i]), 3),
                          round(float(A["d_win"][i]), 3), round(float(A["share"][i]), 3)]
                         for i in np.flatnonzero(mo)]
        # does the network go too far from even? slope of the playouts' chance on its chance
        x, y = A["share"], A["d_win"]
        res["slope"] = float(np.polyfit(x, y, 1)[0])
        res["phases"] = []
        for name, lo, hi in PHASES:
            k = (A["progress"] >= lo) & (A["progress"] < hi)
            km = k & mo
            if k.sum() < 4 or km.sum() < 4:
                continue
            res["phases"].append({"phase": name, "bar": err(A["share"][k], A["d_win"][k], A["d_win_var"][k]),
                                  "margin": err(A["margin"][km], A["d_margin"][km], A["d_margin_var"][km]),
                                  "margin_vs_end": float(np.sqrt(np.mean((A["margin"][km] - A["final"][km]) ** 2))),
                                  "spread": float(np.sqrt(np.mean(A["spread"][km] ** 2)))})
        res["calibration"] = []
        for b in range(10):
            k = (x >= b / 10) & (x < (b + 1) / 10 if b < 9 else x <= 1.0)
            if k.sum():
                res["calibration"].append({"bin": b, "n": int(k.sum()), "net": float(x[k].mean()),
                                           "deep": float(y[k].mean()),
                                           "se": float(np.sqrt(y[k].var() / k.sum() + A["d_win_var"][k].mean() / k.sum()))})
        out[group] = res

    # the coach's verdicts against the deep analysis
    M = []
    for gi, gm in enumerate(games):
        for j, mv in enumerate(gm["moves"]):
            played = deep[(gi, "pos", mv["pos"] + 1)] if mv["pos"] + 1 < len(gm["positions"]) - 1 else None
            if played is None:            # the move ended the game: its outcome is exact
                fin = gm["positions"][-1]["margin"][mv["me"]]
                played = np.full((len(deep[(gi, "alt", j, 0)]) if mv["alts"] else 1,), fin)
            else:
                played = played[:, mv["me"], 0]
            diffs = []
            for a in range(len(mv["alts"])):
                d = deep[(gi, "alt", j, a)][:, mv["me"], 0] - played
                diffs.append((float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))))
            if not diffs:
                continue
            best = max(diffs, key=lambda t: t[0])
            M.append({"group": "2" if gm["players"] == 2 else "3-4", "who": mv["who"],
                      "loss": mv["loss"], "left": mv["left"], "top_played": mv["top_played"],
                      "lead": gm["positions"][mv["pos"]]["margin"][mv["me"]],
                      "deep_top": 0.0 if mv["top_played"] else diffs[0][0],
                      "deep_top_se": 0.0 if mv["top_played"] else diffs[0][1],
                      "deep": max(0.0, best[0]), "deep_se": best[1], "exact": mv.get("exact"),
                      "deep_alts": [d for d, _ in diffs]})
    coach = {}
    for group in ("2", "3-4"):
        for who in ("all", "greedy", "student"):
            st = coach_stats([m for m in M if m["group"] == group and who in ("all", m["who"])])
            if st:
                coach.setdefault(group, {})[who] = st
    if coach:
        out["coach"] = coach
    return out


def coach_stats(M):
    """The coach's points lost (the network's estimate, one turn deep)
    against the deep analysis, for the moves M. Compared in pairs: the
    move played against the network's first choice, both played out with
    the same futures (when the first choice was played, nothing to compare)."""
    K = [m for m in M if not m["top_played"]]
    if len(K) < 3:
        return None
    q, dt = np.array([m["loss"] for m in K]), np.array([m["deep_top"] for m in K])
    three = lambda x: 0 if x < 2.0 else 1 if x < 4.0 else 2        # fine / inaccuracy / mistake or worse
    res = {"moves": len(M), "compared": len(K), "corr": float(np.corrcoef(q, dt)[0, 1]),
           "mae": float(np.mean(np.abs(q - dt))), "bias": float(np.mean(q - dt)),
           "rmse": float(np.sqrt(max(0.0, np.mean((q - dt) ** 2) - np.mean([m["deep_top_se"] ** 2 for m in K])))),
           "slope": float(np.polyfit(q, dt, 1)[0]), "deep_se": float(np.mean([m["deep_top_se"] for m in K])),
           "agree": float(np.mean([three(m["loss"]) == three(m["deep"]) for m in M])),
           "confusion": {}}
    for m in M:
        key = classify(m["loss"]) + ">" + classify(m["deep"])
        res["confusion"][key] = res["confusion"].get(key, 0) + 1
    big = q >= 4.0
    res["called_mistake"] = int(big.sum())
    res["confirmed_2"] = float(np.mean(dt[big] >= 2.0)) if big.any() else None
    res["confirmed_4"] = float(np.mean(dt[big] >= 4.0)) if big.any() else None
    fine = [m for m in M if m["loss"] < 2.0]
    res["called_fine"] = len(fine)
    res["missed_4"] = float(np.mean([m["deep"] >= 4.0 and m["deep"] > 2 * m["deep_se"] for m in fine])) if fine else None
    # does the deep analysis prefer another move than the network's first choice?
    other = [m for m in K if len(m["deep_alts"]) > 1]
    res["other_best"] = float(np.mean([max(m["deep_alts"][1:]) > m["deep_alts"][0] + 2 * m["deep_top_se"]
                                       for m in other])) if other else None
    ex = [m for m in M if m["exact"] is not None]
    if ex:
        res["endgame"] = {"moves": len(ex), "net_mae": float(np.mean([abs(m["loss"] - m["deep"]) for m in ex])),
                          "exact_mae": float(np.mean([abs(m["exact"] - m["deep"]) for m in ex]))}
    for name, sel in (("close", lambda m: abs(m["lead"]) < 15), ("far", lambda m: abs(m["lead"]) >= 15)):
        P = [m for m in K if sel(m)]
        if len(P) > 2:
            qq, dd = np.array([m["loss"] for m in P]), np.array([m["deep_top"] for m in P])
            res[name] = {"moves": len(P), "mae": float(np.mean(np.abs(qq - dd))),
                         "bias": float(np.mean(qq - dd)), "corr": float(np.corrcoef(qq, dd)[0, 1])}
    res["points"] = [[round(m["loss"], 2), round(m["deep_top"], 2), round(m["deep_top_se"], 2),
                      round(m["deep"], 2), m["left"], round(m["lead"], 1)] for m in K]
    return res


def report(s):
    pp = lambda x: "-" if x is None else "%.1f%%" % (100 * x)
    for group in ("2", "3-4"):
        if group not in s:
            continue
        r = s[group]
        print("\n%s players (%d positions):" % (group, r["positions"]))
        print("  chance to win, the bar:    typical difference %s (mean %s; the playouts' own luck %s)"
              % (pp(r["bar"]["rmse"]), pp(r["bar"]["mae"]), pp(r["bar"]["noise"])))
        print("  chance to win, the bubble: typical difference %s (network minus playouts on average %+.1f%%)"
              % (pp(r["bubble"]["rmse"]), 100 * r["bubble"]["bias"]))
        print("  final margin:              typical difference %.1f points (bias %+.1f); against the real end %.1f, "
              "the futures themselves differ by %.1f" % (r["margin"]["rmse"], r["margin"]["bias"],
                                                           r["margin_vs_end"], r["spread"]))
        print("  slope (1 = right, below 1: the network goes too far from even): %.2f" % r["slope"])
        print("  margin slope (above 1: big leads squeezed): %.2f; close positions (|margin| < 15) %.1f points, "
              "big leads %.1f points (bias %+.1f)" % (r["margin_slope"], r["margin_close"]["rmse"],
                                                      r["margin_far"]["rmse"], r["margin_far"]["bias"]))
        for ph in r["phases"]:
            print("    %-7s chance %5s   margin %4.1f points   (vs the real end %4.1f, futures differ %4.1f)"
                  % (ph["phase"], pp(ph["bar"]["rmse"]), ph["margin"]["rmse"], ph["margin_vs_end"], ph["spread"]))
        print("  calibration: " + "  ".join("%s->%s (%d)" % (pp(c["net"]), pp(c["deep"]), c["n"]) for c in r["calibration"]))
    for group, byw in s.get("coach", {}).items():
        for who, c in byw.items():
            print("\nThe coach, %s players, %s (%d moves, %d compared in pairs):"
                  % (group, "all judged moves" if who == "all" else "the " + who, c["moves"], c["compared"]))
            print("  points lost, coach against the deep analysis: typical difference %.1f (without the playouts' "
                  "own luck of ±%.1f), mean difference %.1f, coach minus deep %+.1f, correlation %.2f, slope %.2f"
                  % (c["rmse"], c["deep_se"], c["mae"], c["bias"], c["corr"], c["slope"]))
            print("  same verdict (fine / inaccuracy / mistake or worse): %s" % pp(c["agree"]))
            print("  called a mistake or blunder: %d; of those at least an inaccuracy after playing out: %s, "
                  "still a mistake: %s" % (c["called_mistake"], pp(c["confirmed_2"]), pp(c["confirmed_4"])))
            print("  called fine: %d; of those a clear mistake after playing out: %s"
                  % (c["called_fine"], pp(c["missed_4"])))
            if c["other_best"] is not None:
                print("  another move than the network's first choice clearly better after playing out: %s"
                      % pp(c["other_best"]))
            for name in ("close", "far"):
                if name in c:
                    print("  %s games (%d moves): mean difference %.1f, coach minus deep %+.1f, correlation %.2f"
                          % ("close" if name == "close" else "lopsided", c[name]["moves"], c[name]["mae"],
                             c[name]["bias"], c[name]["corr"]))
            if "endgame" in c:
                e = c["endgame"]
                print("  last 2 turns (%d moves): difference with the deep analysis: network %.1f, exact count %.1f points"
                      % (e["moves"], e["net_mae"], e["exact_mae"]))


def main():
    ap = argparse.ArgumentParser(description="The network's quick estimate against the deep analysis.")
    ap.add_argument("--games2", type=int, default=24, help="2-player games")
    ap.add_argument("--games4", type=int, default=6, help="3-4 player games (half with 3)")
    ap.add_argument("--n2", type=int, default=32, help="playouts per position, 2 players")
    ap.add_argument("--n4", type=int, default=16, help="playouts per position, 3-4 players")
    ap.add_argument("--first-seed", type=int, default=77_000_000)
    ap.add_argument("--workers", type=int, default=4,
                    help="processes for the playouts (each takes about 2 GB of GPU memory)")
    ap.add_argument("--parallel", type=int, default=128, help="games at once per process")
    ap.add_argument("--no-judge4", dest="judge4", action="store_false",
                    help="3-4 players: only the positions, not the moves (quicker)")
    a = ap.parse_args()
    t0 = time.time()
    jobs = []
    for i in range(a.games2):              # a quarter the network against itself, a quarter
        other = ["nn", "greedy", "student", "student"][i % 4]   # against greedy, half against a student
        seats = ["nn", other] if i // 4 % 2 == 0 else [other, "nn"]
        jobs.append((2, seats, a.first_seed + i, other != "nn"))
    for i in range(a.games4):
        n_pl = 3 if i % 2 == 0 else 4
        seats = (["nn", "student", "greedy"] if n_pl == 3 else ["nn", "student", "nn", "greedy"])
        k = i // 2 % n_pl
        jobs.append((n_pl, seats[k:] + seats[:k], a.first_seed + 1000 + i, a.judge4))
    with ProcessPoolExecutor(max(1, min(len(jobs), (os.cpu_count() or 2) - 2))) as ex:
        games = list(ex.map(play_game, jobs))
    n_pos = sum(len(g["positions"]) - 1 for g in games)
    n_mov = sum(len(g["moves"]) for g in games)
    print("%d games played in %.0f s: %d positions, %d moves to judge" % (len(games), time.time() - t0, n_pos, n_mov),
          flush=True)

    # every start of a game in jobs of CHUNK, all with the game's seed (the same futures)
    work = []
    for gi, gm in enumerate(games):
        n = a.n2 if gm["players"] == 2 else a.n4
        starts = [((gi, "pos", i), pos["state"]) for i, pos in enumerate(gm["positions"][:-1])]
        starts += [((gi, "alt", j, k), st) for j, mv in enumerate(gm["moves"]) for k, st in enumerate(mv["alts"])]
        for c in range(0, len(starts), CHUNK):
            part = starts[c:c + CHUNK]
            work.append(([k for k, _ in part], gm["players"], gm["seed"], n, [st for _, st in part]))
    total = sum(len(w[4]) * w[3] for w in work)
    print("%d starts, %d playouts in %d jobs" % (sum(len(w[4]) for w in work), total, len(work)), flush=True)
    deep, done, t1 = {}, 0, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(play_out_job, (i,) + w[1:] + (a.parallel,)): w for i, w in enumerate(work)}
        for f in as_completed(futs):
            w = futs[f]
            _, res = f.result()
            for key, r in zip(w[0], res):
                deep[key] = r
            done += len(w[4]) * w[3]
            el = time.time() - t1
            print("  %d/%d playouts (%.0f per second), about %.0f minutes left"
                  % (done, total, done / el, (total - done) / (done / el) / 60), flush=True)
    s = summarize(games, deep)
    s["made"] = time.strftime("%Y-%m-%d %H:%M")
    s["settings"] = {"games2": a.games2, "games4": a.games4, "n2": a.n2, "n4": a.n4,
                     "models": {str(k): v for k, v in MODELS.items()}, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(ROOT, "results", "deep_check.json"), "w") as f:
        json.dump(s, f, indent=1)
    report(s)
    print("\nDone in %.0f minutes; results/deep_check.json" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
