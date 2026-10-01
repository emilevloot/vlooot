"""
Deep analysis: a move judged by playing the rest of the game out, many times.

The coach (review.py) judges a move by one estimate of the network: what it
thinks the position after the move is worth. That estimate is rough (about
12 points off on average in the middle of a game). Playing the game out is
slower but far more certain: from the position after the move that was
played, and after the best few other moves, the network plays the rest of
the game against itself, many times. Each time the bag of longships is
shuffled (nobody knows which ships come next), so every playout is a
different future. The average final margin of each move, and how often it
won, shows how good the move really was - with a margin of uncertainty.

The moves are compared in pairs: playout k of every move gets the same
future (the same order of longships, the same dice), so the difference
between two moves isn't drowned in the luck of the playouts.

It runs in the fast engine (fastgame.py, mp_game.py for 3-4 players) with
the network on the GPU when there is one, all playouts at once in batches:
server.py (/ai-analyze) for the local game.

    python analysis.py          (a quick test on a game of the network)
"""

import copy
import random
import time
import zlib

import numpy as np

import looot as L
import nn_bot
import review

_NETS = {}


def _engine(n_players):
    if n_players == 2:
        import fastgame
        return fastgame
    import mp_game
    return mp_game


def _net(model):
    """The network for the playouts: on the GPU when there is one."""
    if model not in _NETS:
        net = None
        try:
            import torch
            if torch.cuda.is_available():
                import gpu_net
                net = gpu_net.TorchNet(model)
        except Exception:
            net = None
        _NETS[model] = net or nn_bot.load_net(model)
    return _NETS[model]


def play_out(starts, me, model, n, seed=1, parallel=256):
    """Play every game in `starts` (looot.Game, at the start of a turn) to
    the end n times, the network on every seat, the bag of longships
    shuffled for each playout - for playout k the same way for every start
    (ships in the same order, the same dice). Returns per start an array
    [n, 2]: the final margin of player `me` and its share of the win (1,
    0.5 for a tie of two, 0), row k being playout k. me=None: for every
    player, an array [n, players, 2]."""
    npl = len(starts[0].players)
    F = _engine(npl)
    net = _net(model)
    rng = np.random.default_rng(seed)
    F.seed(seed % (2 ** 31))
    order = [rng.permutation(64) for _ in range(n)]     # playout k: each ship's place in the bag
    jobs = []
    for i, g in enumerate(starts):
        s0, gs = F.from_game(g)
        for r in range(n):
            s = s0.copy()
            nb = int(s[F.S_BAGN])
            if nb > 1:
                bag = s[F.S_BAG:F.S_BAG + nb].copy()
                s[F.S_BAG:F.S_BAG + nb] = bag[np.argsort(order[r][bag], kind="stable")]
            jobs.append((i, r, s, gs))
    out = [np.zeros((n, npl, 2)) for _ in starts]
    todo = list(range(len(jobs)))
    players = []
    slots = []

    def finish(slot):
        i, r = jobs[slot["j"]][:2]
        s = slot["s"]
        out[i][r] = [F.outcome(s, p) if npl == 2 else F.outcome(s, slot["gs"], p) for p in range(npl)]

    def advance(slot, answer):
        """Run a playout until its next question for the network; True when it ended."""
        while True:
            if slot["steps"] is None and slot["s"][F.S_OVER]:
                return True
            try:
                if slot["steps"] is None:
                    slot["steps"] = slot["player"].turn_steps(slot["s"], slot["gs"])
                    slot["req"] = next(slot["steps"])
                else:
                    slot["req"] = slot["steps"].send(answer)
                return False
            except StopIteration as done:
                slot["s"] = done.value
                slot["steps"] = None
                slot["req"] = None
                answer = None

    def start(j, player):
        _, r, s, gs = jobs[j]
        player.rng = random.Random(seed * 100003 + r)       # the same dice for playout r of every move
        return {"j": j, "s": s, "gs": gs, "player": player, "steps": None, "req": None}

    while todo or slots:
        while todo and len(slots) < parallel:
            player = players.pop() if players else F.FastPlayer(net, random.Random(0), 0.0, noise=False, shuffle=False)
            slot = start(todo.pop(0), player)
            if advance(slot, None):
                finish(slot)
                players.append(slot["player"])
                continue
            slots.append(slot)
        if not slots:
            break
        pol = [sl for sl in slots if sl["req"][0] is None]
        val = [sl for sl in slots if sl["req"][0] is not None]
        got = {}
        for sl, ans in zip(pol, F.proposal_maps(net, [(sl["req"][1], sl["gs"], sl["req"][2]) for sl in pol])):
            got[id(sl)] = ans
        for sl, ans in zip(val, F.judge_states(net, [(sl["req"][0], sl["req"][1], sl["gs"], sl["req"][2])
                                                     for sl in val])):
            got[id(sl)] = ans
        keep = []
        for sl in slots:
            if advance(sl, got[id(sl)]):
                finish(sl)
                players.append(sl["player"])
            else:
                keep.append(sl)
        slots = keep
    return out if me is None else [o[:, me] for o in out]


def same_move(ga, gb, me):
    """Did player `me` end the turn in the same way in both games?"""
    pa, pb = ga.players[me], gb.players[me]
    return (pa.fjord == pb.fjord and pa.trophy == pb.trophy and pa.shields == pb.shields
            and {c: v.count(me) for c, v in ga.vikings.items()} == {c: v.count(me) for c, v in gb.vikings.items()})


def candidates(model, g0):
    """The moves the network considers for the player to move in g0 (as
    the coach does): the games after them and the network's final margin
    for each, best first."""
    bot = nn_bot.NNBot(random.Random(0), model)
    cands, vals = bot._candidates(copy.deepcopy(g0))
    order = np.argsort(-vals)
    return [cands[k][5] for k in order], vals[order]


def alternatives(g0, g1, games, vals, alts=3, lang="en"):
    """The best `alts` of the candidates (games, vals: candidates()) other
    than the move from g0 to g1: [(game after it, describe(), value)]."""
    me = g0.current
    chosen, texts = [], {review.describe(g0, g1, me, lang)["text"]}
    for g, v in zip(games, vals):
        if same_move(g, g1, me):
            continue
        d = review.describe(g0, g, me, lang)
        if d["text"] in texts:                      # the same move in other words (fjord spaces)
            continue
        texts.add(d["text"])
        chosen.append((g, d, float(v)))
        if len(chosen) >= alts:
            break
    return chosen


def analyse_turn(model, before, after, n=16, alts=3, lang="en"):
    """Play the move from `before` to `after`, and the best `alts` other
    moves the network finds, out n times each. Returns the moves, best
    first, with their average final margin, its uncertainty and the share
    of playouts won, and how many points the move cost against the best."""
    t0 = time.time()
    g0 = L.Game.load_state(before)
    g1 = L.Game.load_state(after)
    me = g0.current
    chosen = alternatives(g0, g1, *candidates(model, g0), alts=alts, lang=lang)
    played = review.describe(g0, g1, me, lang)
    starts = [g1] + [g for g, _, _ in chosen]
    res = play_out(starts, me, model, n, seed=zlib.crc32(str(before.get("seed")).encode()) % 100000 + g0.turn_no)
    moves = []
    for i, (r, (g, d)) in enumerate(zip(res, [(g1, played)] + [(g, d) for g, d, _ in chosen])):
        diff = r[:, 0] - res[0][:, 0]                   # against the move played, playout by playout
        moves.append({"played": i == 0, "text": d["text"], "cells": d["cells"], "fjord": d["fjord"],
                      "mean": round(float(r[:, 0].mean()), 2), "win": round(float(r[:, 1].mean()), 3),
                      "vs_played": round(float(diff.mean()), 2),
                      "se": round(float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, 2)})
    best = max(moves, key=lambda m: m["vs_played"])
    loss = max(0.0, best["vs_played"])
    se = best["se"]
    moves.sort(key=lambda m: -m["mean"])
    return {"n": n, "who": me, "turn": g0.turn_no, "moves": moves,
            "loss": round(loss, 2), "loss_se": round(se, 2),
            "cls": review.classify(loss, 0.0, 0.0), "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    lay = L.read_layout_file()
    g = L.Game([("a", True), ("b", True)], 5, lay)
    bot = nn_bot.NNBot(random.Random(5), "s2_r32")
    for _ in range(9):
        bot.play_turn(g)
    before = review.snapshot(g)
    L.Bot(random.Random(1)).play_turn(g)               # a greedy move, to judge
    after = review.snapshot(g)
    for n in (16, 32):
        r = analyse_turn("s2_r32", before, after, n=n)
        print("n=%d: %.1f s; the move cost %.1f points (±%.1f) -> %s"
              % (n, r["secs"], r["loss"], 1.96 * r["loss_se"], r["cls"]))
        for m in r["moves"]:
            print("   %s %+6.1f  (%+5.1f ±%4.1f against the move played)  won %3.0f%%  %s"
                  % ("*" if m["played"] else " ", m["mean"], m["vs_played"], 1.96 * m["se"], 100 * m["win"], m["text"]))
