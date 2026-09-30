"""
The end of the game, worked out exactly instead of guessed.

Near the end the network's judgement is rough (about 7 points off with 1-5
Vikings left), while there a move is often only worth a few points. But the
end can simply be counted: once a player has no Vikings left, their fjord
and trophy never change again, so their final score is known.

reach(g) looks at the player to move, when it has at most 2 turns left:
every placement (with every shield), many more fjord layouts than the
network tries, every longship on many spaces, and the trophy; for 2 turns
left, the best few ways through this turn, each followed by the best last
turn. What the others do in between is left out: they can take a space you
wanted, but they can't change your fjord.

  - the player (NNBot with endgame=True) uses it in its last turn;
  - the coach (review.py) and the dashboard's check (selfplay_record.py)
    use it to find endgame mistakes exactly: the best final score the
    player could still reach before its move, against what it could still
    reach after it.
"""

import copy
import random

import numpy as np

import nn_bot

WIDE = dict(TILE_K=8, MAX_LAYOUTS=40, SHIP_K=8, TOP_FULL=24)
NARROW = dict(TILE_K=4, MAX_LAYOUTS=10, SHIP_K=4, TOP_FULL=10)
BEAM = 4                # with 2 turns left: the best ways through this turn tried further


class _Solver(nn_bot.NNBot):
    """NNBot's move generator, wider and without the proposals. A finished
    turn after which the player has no Vikings left is judged by its own
    final score (exact); other turns by the network, as before."""

    def __init__(self, net_name, width):
        super().__init__(random.Random(0), net_name)
        self.use_policy = False
        for k, v in width.items():
            setattr(self, k, v)

    def _judge(self, games, me):
        vals = np.zeros(len(games))
        todo = []
        for k, g in enumerate(games):
            if g.players[me].vikings_left == 0:
                vals[k] = g.players[me].score()["total"]
            else:
                todo.append(k)
        if todo:
            vals[todo] = super()._judge([games[k] for k in todo], me)
        return vals


def again(g, me):
    """The game after a turn, with `me` to move again right away (the others
    skipped), or None when `me` has no turn left."""
    p = g.players[me]
    if g.game_over or p.vikings_left <= 0:
        return None
    g2 = copy.deepcopy(g)
    g2.current = me
    g2.new_turn()
    return g2


def reach(g, net_name, width=WIDE, beam=BEAM, tie_break=False):
    """The best final score the player to move can still reach, if it has at
    most 2 turns left: (score, candidate, all candidates), where a candidate
    is NNBot's (placement, tiles, ..., finished copy); None otherwise.
    tie_break: of the moves that reach the same best score, the one the
    network likes best (it may leave the opponent less, say)."""
    me = g.current
    p = g.players[me]
    if g.game_over or not 1 <= p.vikings_left <= 2:
        return None
    last = p.vikings_left == 1
    solver = _Solver(net_name, width if last else NARROW)
    cands, vals = solver._candidates(copy.deepcopy(g))
    if not cands:
        return None
    best = None
    later = []
    for k, c in enumerate(cands):
        if c[5].players[me].vikings_left == 0:
            v = float(vals[k])
            if best is None or v > best[0]:
                best = (v, k)
        else:
            later.append(k)
    # with a turn still to come: the best few (by the network), each with its best last turn
    for k in sorted(later, key=lambda k: -vals[k])[:beam]:
        g2 = again(cands[k][5], me)
        r = reach(g2, net_name, width, beam) if g2 is not None else None
        if r is not None and (best is None or r[0] > best[0]):
            best = (r[0], k)
    if best is None:
        return None
    if tie_break and best[1] not in later:
        ties = [k for k in range(len(cands)) if k not in later and vals[k] >= best[0] - 1e-6]
        if len(ties) > 1:
            v = nn_bot.NNBot._judge(solver, [cands[k][5] for k in ties], me)
            best = (best[0], ties[int(np.argmax(v))])
    return best[0], cands[best[1]], cands


def after_move(g_after, me, net_name, width=WIDE, beam=BEAM):
    """The best final score `me` can still reach after its move: exact when
    it has no Vikings left, else its best remaining turn (the others skipped)."""
    p = g_after.players[me]
    if p.vikings_left == 0:
        return float(p.score()["total"])
    g2 = again(g_after, me)
    r = reach(g2, net_name, width, beam) if g2 is not None else None
    return r[0] if r is not None else None


def check(before, after, me, net_name, width=WIDE):
    """How many points of final score a move threw away: the best the player
    could reach before it, minus the best it can reach after it. Returns
    {"best", "played", "loss", "best_move": finished copy} or None when the
    player had more than 2 turns left."""
    r = reach(before, net_name, width)
    if r is None:
        return None
    played = after_move(after, me, net_name, width)
    if played is None:
        return None
    return {"best": r[0], "played": played, "loss": max(0.0, r[0] - played),
            "best_move": r[1]}
