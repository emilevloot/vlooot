"""
Test arena for Looot computer players.

Plays many games between computer players and reports who wins, by how
much, where their points come from and how long they think.

    python arena.py greedy random random
    python arena.py greedy greedy --games 300
    python arena.py greedy random --games 100 --random-boards

The names are the players at the table (2 to 4), in seat order. Seats are
rotated every game, and each group of rotations uses the same boards and
longship bag, so no player profits from a lucky seat or a lucky draw.

To add a player: write a class with a play_turn(game) method (see Bot in
looot.py), then add it to PLAYERS below.
"""

import os
# One math thread per process: the arena already runs a process per core,
# and extra numpy threads would only fight over the same cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import looot as L


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class RandomBot:
    """Plays legal but random moves: a random Viking spot, random fjord
    spots for its tiles, no longships or shields. A baseline to beat."""

    def __init__(self, rng):
        self.rng = rng

    def play_turn(self, g):
        p = g.player()
        if g.phase == "place":
            opts = g.legal_cells(False)
            if opts:
                g.place_viking(self.rng.choice(opts))
                while g.phase == "tiles":
                    g.place_tile(0, self.rng.choice(p.empty_cells()))
        cl = g.claimable_trophies(p)
        if cl:
            g.claim_trophy(max(cl))
        if g.phase == "actions":
            g.end_turn()


# The greedy player's hand-picked numbers from before tuning, kept so the
# arena can show how much the tuned BOT_WEIGHTS gained.
ORIGINAL_WEIGHTS = {
    "ship_base": 0.95, "ship_per_missing": 0.22, "ship_late": 0.25,
    "ship_min": 0.10, "ship_hopeless": 1.3, "ship_hopeless_chance": 0.05,
    "ship_future_res": 0.30, "ship_future_bld": 0.15,
    "site_power": 2.0, "site_early": 0.9, "site_late": 0.4, "site_late_left": 2,
    "trophy_ready": 0.9, "trophy_progress": 0.5,
    "shield_early": 1.5, "shield_late": 0.3, "shield_late_left": 3,
    "chain": 0.8, "tile_need": 10.0, "tile_block": 3.0, "tile_open": 0.2,
    "shipcell_match": 5.0, "shipcell_room": 1.0,
    "extra_gain": 3.0, "extra_force_left": 3, "take_ship_margin": 1.0,
    "trophy_wait_left": 2, "noise": 0.3,
}

# name -> function that makes a player from a random generator
PLAYERS = {
    "random": RandomBot,
    "greedy": L.Bot,                                    # uses BOT_WEIGHTS
    "original": lambda rng: L.Bot(rng, ORIGINAL_WEIGHTS),
}


def make_player(spec, rng):
    """spec is a name from PLAYERS, or (name, weights) for a greedy
    player with its own weights (used by tune.py)."""
    if isinstance(spec, tuple):
        return L.Bot(rng, spec[1])
    if spec.startswith("nn:"):                  # a trained network, see nn_bot.py
        import nn_bot
        return nn_bot.player_from_name(spec, rng)
    return PLAYERS[spec](rng)


def spec_name(spec):
    return spec[0] if isinstance(spec, tuple) else spec

SCORE_PARTS = ["castle", "watchtower", "house", "gold", "sheep", "wood"]


# ---------------------------------------------------------------------------
# Playing games
# ---------------------------------------------------------------------------

def play_game(lineup, seed, rotation, layout, rules=None):
    """Play one game. lineup: player names in seat order before rotating.
    rules: a stage name from looot.RULE_STAGES, or None for the full game.
    Returns one result dict per seat."""
    n = len(lineup)
    specs = [lineup[(k + rotation) % n] for k in range(n)]
    seats = [spec_name(s) for s in specs]
    g = L.Game([("%s#%d" % (name, k), True) for k, name in enumerate(seats)],
               seed, layout, L.rules_for(rules))
    bots = [make_player(spec, random.Random(seed * 1009 + 17 * k + rotation))
            for k, spec in enumerate(specs)]
    think = [0.0] * n
    turns = [0] * n
    worst = [0.0] * n
    moves = 0
    while not g.game_over:
        k = g.current
        t = time.perf_counter()
        bots[k].play_turn(g)
        dt = time.perf_counter() - t
        think[k] += dt
        worst[k] = max(worst[k], dt)
        turns[k] += 1
        moves += 1
        if moves > 500:
            raise RuntimeError("game %d did not finish (a player is stuck)" % seed)
    winners = g.winners()
    scores = [p.score() for p in g.players]
    totals = [s["total"] for s in scores]
    out = []
    for k in range(n):
        best_other = max(totals[j] for j in range(n) if j != k)
        parts = {c: scores[k]["rows"][c]["points"] for c in SCORE_PARTS}
        parts.update(sites=scores[k]["sites"], trophy=scores[k]["trophy"],
                     penalty=scores[k]["penalty"])
        out.append({
            "name": seats[k], "seat": k,
            "win": (1.0 / len(winners)) if k in winners else 0.0,
            "score": totals[k], "margin": totals[k] - best_other,
            "parts": parts,
            "think": think[k], "turns": turns[k], "worst": worst[k],
        })
    return out


def _job(args):
    return play_game(*args)


def run(lineup, games, seed=1, layout=None, workers=None, pool=None, rules=None):
    """Play `games` games. Pass an open ProcessPoolExecutor as `pool` to
    reuse it between calls (tune.py does that)."""
    n = len(lineup)
    jobs = [(lineup, seed + i // n, i % n, layout, rules) for i in range(games)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunk = max(1, games // (workers * 4))
    if pool is not None:
        return list(pool.map(_job, jobs, chunksize=chunk))
    if workers == 1:
        return [play_game(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs, chunksize=chunk))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def ci95(values):
    """Mean and the +/- of its 95% confidence interval."""
    m = statistics.mean(values)
    if len(values) < 2:
        return m, float("nan")
    return m, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def report(lineup, results, seconds):
    n = len(lineup)
    rows = [r for game in results for r in game]
    lineup = [spec_name(s) for s in lineup]
    names = sorted(set(lineup), key=lineup.index)
    fair = 1.0 / n
    print()
    print("%d games, %d players: %s   (%.0f s)" %
          (len(results), n, " vs ".join(lineup), seconds))
    print("A fair share of wins is %.0f%% per seat." % (100 * fair))
    print()
    print("%-10s %6s %15s %13s %14s %9s %9s" %
          ("player", "seats", "win rate", "avg score", "avg margin",
           "ms/turn", "max ms"))
    for name in names:
        rs = [r for r in rows if r["name"] == name]
        w, wci = ci95([r["win"] for r in rs])
        s, sci = ci95([r["score"] for r in rs])
        m, mci = ci95([r["margin"] for r in rs])
        ms = 1000 * sum(r["think"] for r in rs) / max(1, sum(r["turns"] for r in rs))
        worst = 1000 * max(r["worst"] for r in rs)
        print("%-10s %6d %8.1f%% ±%4.1f %7.1f ±%4.1f %+8.1f ±%4.1f %9.1f %9.0f" %
              (name, len(rs), 100 * w, 100 * wci, s, sci, m, mci, ms, worst))
    print()
    print("Where the points come from (average per game):")
    keys = SCORE_PARTS + ["sites", "trophy", "penalty"]
    print("%-10s " % "" + " ".join("%7s" % k[:7] for k in keys))
    for name in names:
        rs = [r for r in rows if r["name"] == name]
        print("%-10s " % name + " ".join(
            "%7.1f" % statistics.mean(r["parts"][k] for r in rs) for k in keys))
    print()
    print("Win rate by seat (all players together), to spot a seat advantage:")
    print("  " + "   ".join(
        "seat %d: %.0f%%" % (k + 1, 100 * statistics.mean(
            r["win"] for r in rows if r["seat"] == k)) for k in range(n)))
    print()
    if len(names) == 2:
        a, b = names
        wa, ca = ci95([r["win"] for r in rows if r["name"] == a])
        share = wa / (lineup.count(a) * fair)
        verdict = ("clearly better" if wa - ca > lineup.count(a) * fair else
                   "clearly worse" if wa + ca < lineup.count(a) * fair else
                   "not clearly different (play more games)")
        print("Verdict: %s is %s than %s (%.2fx its fair share of wins)." %
              (a, verdict, b, share))


def main():
    ap = argparse.ArgumentParser(description="Play computer players against each other.")
    ap.add_argument("players", nargs="+",
                    help="2 to 4 players: %s, or nn:<model> for a trained "
                         "network (2 players only), e.g. greedy random random"
                         % ", ".join(sorted(PLAYERS)))
    ap.add_argument("--games", type=int, default=120,
                    help="number of games (default 120)")
    ap.add_argument("--seed", type=int, default=1, help="first seed (default 1)")
    ap.add_argument("--random-boards", action="store_true",
                    help="random boards instead of boards.json")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (default: all cores but one)")
    ap.add_argument("--rules", default=None,
                    choices=[s for s, _ in L.RULE_STAGES],
                    help="play with only some rules (default: the full game)")
    a = ap.parse_args()
    if not 2 <= len(a.players) <= 4:
        ap.error("give 2 to 4 players")
    for p in a.players:
        if p not in PLAYERS and not p.startswith("nn:"):
            ap.error("unknown player %r" % p)
    layout = None if a.random_boards else L.read_layout_file()
    t = time.time()
    results = run(a.players, a.games, a.seed, layout, a.workers or None,
                  rules=a.rules)
    if a.rules:
        print("\nRules: stage '%s' only." % a.rules)
    report(a.players, results, time.time() - t)


if __name__ == "__main__":
    sys.exit(main())
