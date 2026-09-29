"""
Train a network by playing against itself only, the AlphaZero way: there is
always one CHAMPION. Every round:

  1. the champion plays itself (--games games, full rules)
  2. a challenger is trained from the champion on the latest self-play
     games only (no greedy games any more), with TD targets
  3. the challenger plays the champion (--match-games games, both seat
     orders on the same boards). Only if it wins more than --accept of them
     does it become the new champion.

So the network has to beat ITS OWN best version to move on; the greedy
player is no longer the yardstick (it only gets a look each round, for the
graphs). Everything goes to curriculum_<prefix>_log.txt / _results.json,
in the curriculum's format (so compare_runs.py and the report work too).

    python selfplay_train.py --prefix s1 --init c5t_full_r10 --hours 3
"""

import argparse
import time

import curriculum as C
import looot as L


def match(a, b, games, seed, players=2):
    """Share of the games network a wins against network b (draws count half).
    With 4 players: a, b, a, b at one table (a's two seats together)."""
    import arena
    seats = ["nn:" + a, "nn:" + b] * (players // 2)
    res = arena.run(seats, games, seed, L.read_layout_file())
    per_game = [sum(r["win"] for r in g if r["name"] == "nn:" + a) for g in res]
    rows = [r for g in res for r in g if r["name"] == "nn:" + a]
    n = len(per_game)
    win = sum(per_game) / n
    return {"games": games, "win": win, "win_ci": 1.96 * (win * (1 - win) / n) ** 0.5,
            "margin": sum(r["margin"] for r in rows) / len(rows)}


def main():
    ap = argparse.ArgumentParser(description="Self-play training with a champion.")
    ap.add_argument("--prefix", default="s1")
    ap.add_argument("--init", required=True, help="the first champion (a models/ name)")
    ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--games", type=int, default=8000, help="self-play games per round")
    ap.add_argument("--window", type=int, default=4, help="train on the last N self-play sets")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--td", type=int, default=2)
    ap.add_argument("--match-games", type=int, default=400)
    ap.add_argument("--accept", type=float, default=0.53,
                    help="the challenger must win more than this share of the match")
    ap.add_argument("--arena-games", type=int, default=300, help="games against greedy per round")
    ap.add_argument("--players", default="2",
                    help="players per game, e.g. 3,4: the self-play games are split over "
                         "these (a 2-4-player network: transfer_mp.py); the match is played "
                         "at a table of 4 (challenger, champion, challenger, champion) and the "
                         "test is against 3 greedy players")
    ap.add_argument("--batch", default="4096", help="positions per training step")
    a = ap.parse_args()
    counts = [int(x) for x in a.players.split(",")]
    table = 4 if max(counts) > 2 else 2
    C.PLAYERS = table                     # the tests against greedy: 1 network, the rest greedy
    deadline = time.time() + 3600 * a.hours

    R = C.Run(a.prefix)
    C.keep_awake()
    st = R.results["stages"].setdefault("full", {"steps": []})
    R.results["selfplay"] = {"init": a.init, "accept": a.accept}
    champion = st.get("champion", a.init)
    rounds = [s for s in st["steps"] if s["kind"].startswith("challenger")]
    k = len(rounds)
    sp = ["%s_sp%d" % (a.prefix, i) for i in range(1, k + 1)]
    R.log("=== self-play with a champion: starting from %s ===" % champion)
    round_time = 0
    while True:
        left = deadline - time.time()
        if left < round_time + C.FINALS_RESERVE:
            R.log("  time budget reached: no more rounds (%.0f min left)" % (left / 60))
            break
        t0 = time.time()
        k += 1
        name = "%s_sp%d" % (a.prefix, k)
        t1 = 0
        for j, npl in enumerate(counts):          # the games split over the player counts
            _, t = R.run(["gen_data.py", "--games", str(a.games // len(counts)), "--player",
                          "nn:" + champion, "--players", str(npl),
                          "--name", name if len(counts) == 1 else "%s_p%d" % (name, npl),
                          "--seed", str(70_000_000 + 100_000 * k + 10_000 * j), "--batch", "750"])
            t1 += t
        sp.append(name)
        new = "%s_r%d" % (a.prefix, k)
        out, t2 = R.run(["train_nn.py", "--data"] + sp[-a.window:] +
                        ["--name", new, "--epochs", str(a.epochs), "--init", champion,
                         "--lr", a.lr, "--td", str(a.td), "--td-mix", "0.5", "--batch", a.batch])
        m = match(new, champion, a.match_games, 80_000_000 + 1000 * k, table)
        g = C.arena_vs_greedy(new, None, a.arena_games, 81_000_000 + 1000 * k)
        won = m["win"] > a.accept
        st["steps"].append({"model": new, "kind": "challenger %d" % k, "test_mae": C.kept(out),
                            "arena": g, "vs_champion": m, "champion": champion, "accepted": won})
        R.log("  round %d: %d self-play games in %.0f s, trained %s in %.0f s" % (k, a.games, t1, new, t2))
        R.log("  %-18s vs champion %s: %5.1f%% ±%.1f (margin %+.1f) -> %s; vs greedy %5.1f%%"
              % (new, champion, 100 * m["win"], 100 * m["win_ci"], m["margin"],
                 "NEW CHAMPION" if won else "stays challenger", 100 * g["win"]))
        if won:
            champion = new
        st["champion"] = champion
        R.save()
        round_time = time.time() - t0

    st["best"] = champion
    R.save()
    R.log("  champion: %s" % champion)
    C.finals(R, argparse.Namespace(arena_games=a.arena_games, arena_games_long=500))
    if champion != a.init:
        m = match(champion, a.init, 500, 88_500_000, table)
        R.results["final"]["%s vs %s (the start)" % (champion, a.init)] = m
        R.save()
        R.log("FINAL %s vs %s (where it started): %.1f%% ±%.1f, margin %+.1f"
              % (champion, a.init, 100 * m["win"], 100 * m["win_ci"], m["margin"]))
    R.log("Self-play training finished.")


if __name__ == "__main__":
    main()
