"""
Make training data for the neural network: play many 2-player games and save
every position together with how the game ended.

    python gen_data.py --games 20000

Each game is played by two greedy players (the tuned computer player). To
show the network more kinds of positions than greedy players normally
reach, each turn has a small chance (--explore) of a random Viking
placement instead of the best one ("exploration").

Every position at the start of a turn is saved twice: once as seen by each
player, labelled with that player's final result:
  margin  = my final score minus the opponent's
  win     = 1 (won), 0.5 (shared) or 0 (lost)

Output: data/<name>_NNN.npz files, one per batch of games.
"""

import os
# One math thread per process (we already run a process per core).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

import looot as L
import nn_encode as E

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


class ExploringBot(L.Bot):
    """The greedy player, but with chance `explore` per turn of placing its
    Viking on a random legal space (the rest of the turn stays greedy)."""

    def __init__(self, rng, explore):
        super().__init__(rng)
        self.explore = explore

    def choose_placement(self, g):
        if self.rng.random() < self.explore:
            opts = self.placement_options(g)
            if opts:
                opt = self.rng.choice(opts)
                g2 = self.simulate_place(g, opt)
                return opt, self.evaluate(g2, g2.players[g.current])
        return super().choose_placement(g)


def make_player(name, rng, explore):
    """Which player makes the moves. 'greedy' for now; later the network
    player can play here too (self-play)."""
    if name == "greedy":
        return ExploringBot(rng, explore)
    import nn_bot
    return nn_bot.player_from_name(name, rng, explore * 0.5)   # it tries more per turn


def on_random_boards(seed, share):
    """Is game `seed` played on newly made random boards (instead of the
    boards of boards.json)? The same answer for the same seed every time."""
    return random.Random(seed * 7919 + 17).random() < share


def uses_fastgame(player):
    """Self-play of a plain network player ('nn:<model>') runs in fastgame.py,
    the Numba copy of the game (same rules and same player, much faster).
    Players with extra options ('@' longship threshold, '+2' look-ahead)
    and the greedy player run in looot.py."""
    return (player.startswith("nn:") and "@" not in player and "+" not in player
            and os.environ.get("LOOOT_ENGINE") != "python")


_TORCH_NETS = {}


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def play_batch_on_gpu(args):
    """A group of self-play games in one process, played in step with the
    network on the GPU (fastgame.selfplay_batched). Returns their data."""
    seeds, explore, layout, player, rules, parallel, players, share = args
    import gpu_net
    name = player.split(":", 1)[1]
    if name not in _TORCH_NETS:
        _TORCH_NETS[name] = gpu_net.TorchNet(name)
    if players == 2:
        import fastgame as engine
        kw = {}
    else:
        import mp_game as engine
        kw = {"players": players}
    out = []
    # the games on the real boards, then those on new random boards
    for lay, part in ((layout, [s for s in seeds if not on_random_boards(s, share)]),
                      (None, [s for s in seeds if on_random_boards(s, share)])):
        if part:
            out += list(engine.selfplay_batched(part, lay, rules, _TORCH_NETS[name],
                                                explore=explore * 0.5, parallel=parallel, **kw))
    return out


def play_and_record(args):
    """Play one game; return the encoded positions and their labels."""
    seed, explore, layout, player, rules, players = args
    if uses_fastgame(player):
        import nn_bot
        net = nn_bot.load_net(player.split(":", 1)[1])
        if players == 2:
            import fastgame
            return fastgame.selfplay_game(seed, layout, rules, net, explore=explore * 0.5)
        import mp_game
        return mp_game.selfplay_game(seed, layout, rules, net, explore=explore * 0.5,
                                     players=players)
    rng = random.Random(seed)
    enc = E
    if players != 2:                # 3-4 players: the multi-player encoding
        import mp_encode as enc
    g = L.Game([("p%d" % k, True) for k in range(players)], seed, layout, L.rules_for(rules))
    bots = [make_player(player, random.Random(seed * 7 + k), explore) for k in range(players)]
    states = []                     # (land, fjord, glob, perspective)
    moves = 0
    while True:
        for me in range(players):
            states.append(enc.encode(g, me) + (me,))
        if g.game_over:
            break
        bots[g.current].play_turn(g)
        moves += 1
        assert moves < 400
    labels = [enc.outcome(g, me) for me in range(players)]
    land = np.stack([s[0] for s in states])
    fjord = np.stack([s[1] for s in states])
    glob = np.stack([s[2] for s in states])
    persp = np.array([s[3] for s in states])
    margin = np.array([labels[p][0] for p in persp], dtype=np.float32)
    win = np.array([labels[p][1] for p in persp], dtype=np.float32)
    game = np.full(len(states), seed, dtype=np.int64)
    del rng
    return land, fjord, glob, margin, win, game


def main():
    ap = argparse.ArgumentParser(description="Make training data (2-player games).")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--explore", type=float, default=0.15,
                    help="chance per turn of a random Viking placement")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="greedy", help="file name prefix")
    ap.add_argument("--player", default="greedy",
                    help="who plays: greedy, or a network file for self-play")
    ap.add_argument("--random-boards", action="store_true", help="only random boards")
    ap.add_argument("--random-share", type=float, default=0.5,
                    help="share of the games on newly made random boards, so a network "
                         "can't learn the boards by heart (the rest: boards.json)")
    ap.add_argument("--batch", type=int, default=1000, help="games per file")
    ap.add_argument("--rules", default=None,
                    choices=[s for s, _ in L.RULE_STAGES],
                    help="play with only some rules (default: the full game)")
    ap.add_argument("--cpu", action="store_true",
                    help="network self-play on the CPU even if there is a GPU")
    ap.add_argument("--gpu-procs", type=int, default=6,
                    help="GPU self-play: processes (each plays --parallel games)")
    ap.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                    help="players per game (3-4: the multi-player encoding, mp_encode.py)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="GPU self-play: games per process played in step")
    a = ap.parse_args()
    if a.parallel is None:          # 4-player positions are bigger: fewer at once
        a.parallel = 64 if a.players == 2 else 32

    os.makedirs(DATA_DIR, exist_ok=True)
    layout = None if a.random_boards else L.read_layout_file()
    share = 1.0 if (a.random_boards or layout is None) else a.random_share
    t0 = time.time()
    on_gpu = uses_fastgame(a.player) and gpu_available() and not a.cpu
    if on_gpu:
        # a few processes, each playing `parallel` games in step and asking
        # the GPU about all their positions at once
        workers = a.gpu_procs
        size = 128
        groups = [list(range(a.seed + i, a.seed + min(i + size, a.games)))
                  for i in range(0, a.games, size)]
        jobs = [(g, a.explore, layout, a.player, a.rules, a.parallel, a.players, share)
                for g in groups]
        func, chunks = play_batch_on_gpu, 1
        print("Self-play on the GPU: %d processes x %d games at once" % (workers, a.parallel),
              flush=True)
    else:
        workers = max(1, (os.cpu_count() or 2) - 1)
        jobs = [(a.seed + i, a.explore, None if on_random_boards(a.seed + i, share) else layout,
                 a.player, a.rules, a.players) for i in range(a.games)]
        func, chunks = play_and_record, 4
    # The workers keep playing; each full batch of games is compressed and
    # written by a separate thread in the meantime.
    saver = ThreadPoolExecutor(max_workers=1)
    saves = []
    buf = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(func, jobs, chunksize=chunks):
            for part in (result if on_gpu else [result]):
                buf.append(part)
                done += 1
                if len(buf) == a.batch or done == a.games:
                    out = os.path.join(DATA_DIR, "%s_%03d.npz" % (a.name, len(saves)))
                    saves.append(saver.submit(save_batch, out, buf))
                    rate = done / (time.time() - t0)
                    print("%d/%d games (%.0f games/s, %d positions for %s), about %.0f s left"
                          % (done, a.games, rate, sum(len(p[3]) for p in buf),
                             os.path.basename(out), (a.games - done) / rate), flush=True)
                    buf = []
    for f in saves:
        f.result()                   # wait for the last files (and show any error)
    saver.shutdown()
    print("Done in %.0f s." % (time.time() - t0))


POLICY_KEYS = ("pol_place", "pol_tile", "pol_ship", "pol_on")


def save_batch(out, parts):
    arrays = dict(
        land=np.concatenate([p[0] for p in parts]),
        fjord=np.concatenate([p[1] for p in parts]),
        glob=np.concatenate([p[2] for p in parts]),
        margin=np.concatenate([p[3] for p in parts]),
        win=np.concatenate([p[4] for p in parts]),
        game=np.concatenate([p[5] for p in parts]))
    if all(len(p) == 10 for p in parts):       # network self-play: what the proposals learn
        for j, k in enumerate(POLICY_KEYS):
            arrays[k] = np.concatenate([p[6 + j] for p in parts])
    np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    main()
