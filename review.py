"""
The coach: judges a finished turn the way a chess trainer does.

For the turn of player `me` (the game before and after it, as saved by
Game.save_state) the network lists the moves it would consider itself
(NNBot._candidates: every placement, fjord layout, longship and trophy,
shortlisted) and says where each one ends, in points of final margin.
The move that was played is judged the same way. The difference with the
best move is the points lost, and that gives the label:

    brilliant   better than every move the network found itself
    great       the best move, and the only good one (the next best
                Viking space is clearly worse)
    best        (nearly) the network's own choice
    excellent / good / inaccuracy / mistake / blunder: more points lost

Accuracy per move falls off with the points lost; a player's accuracy is
the average over their moves. It is one turn deep: the network's own
judgement, not the truth.

Used by server.py (/ai-review) and, on the web site, by the page itself
(Api.nn_review in looot.py).
"""

import copy
import json
import math
import random

import numpy as np

import looot as L
import nn_bot

# (label, most points lost) - checked in this order
CLASSES = [("best", 0.6), ("excellent", 1.5), ("good", 3.0),
           ("inaccuracy", 5.0), ("mistake", 8.0), ("blunder", float("inf"))]
BRILLIANT = 1.0      # points better than the network's own best move
GREAT_GAP = 5.0      # ... and the next best space this much worse: the only good move
ACC_SCALE = 8.0      # accuracy = 100 * exp(-points lost / ACC_SCALE)

ITEMS = ["wood", "sheep", "gold", "axe", "house", "watchtower", "castle"]
WORDS = {
    "en": {"forest": "Forest", "field": "Field", "mountain": "Mountain", "battlefield": "Battlefield",
           "house": "House", "watchtower": "Watchtower", "castle": "Castle",
           "wood": "wood", "sheep": "sheep", "gold": "gold", "axe": "axe",
           "stack": "stacked", "double": "double", "none": "no Viking",
           "ship": "longship", "trophy": "trophy", "vp": "VP", "lost": "lost"},
    "nl": {"forest": "bos", "field": "weide", "mountain": "berg", "battlefield": "slagveld",
           "house": "huis", "watchtower": "wachttoren", "castle": "kasteel",
           "wood": "hout", "sheep": "schaap", "gold": "goud", "axe": "bijl",
           "stack": "bezet, schild", "double": "dubbel, schild", "none": "geen Viking",
           "ship": "schip", "trophy": "trofee", "vp": "punten", "lost": "kwijt"},
}
ITEM_WORD = {"en": {"house": "house", "watchtower": "watchtower", "castle": "castle"},
             "nl": {"house": "huis", "watchtower": "wachttoren", "castle": "kasteel"}}


def snapshot(g):
    """g.save_state() shares lists with the game: a real copy."""
    return json.loads(json.dumps(g.save_state()))


def counts(p):
    c = dict.fromkeys(ITEMS, 0)
    for it in p.fjord.values():
        if it["kind"] in ("res", "bld"):
            c[it["type"]] += 1
    return c


def describe(g0, g1, me, lang="en"):
    """What player `me` did between g0 and g1, in words, plus the Viking
    spaces used (for rings on the board)."""
    W = WORDS[lang]
    word = lambda t: ITEM_WORD[lang].get(t, W[t])
    p0, p1 = g0.players[me], g1.players[me]
    cells, parts = [], []
    for c, v in g1.vikings.items():
        n = v.count(me) - g0.vikings.get(c, []).count(me)
        if n > 0:
            cells.append(L.key(c))
            tags = []
            if p0.shields["occupy"] and not p1.shields["occupy"] and len(v) > 1:
                tags.append(W["stack"])
            parts.append(W[g1.land[c]] + (" (%s)" % ", ".join(tags) if tags else ""))
    if p0.shields["double"] and not p1.shields["double"] and parts:
        parts[0] += " (%s)" % W["double"]
    text = " + ".join(parts) if parts else W["none"]
    c0, c1 = counts(p0), counts(p1)
    got = {k: c1[k] - c0[k] for k in ITEMS if c1[k] != c0[k]}
    if got:
        text += " → " + ", ".join("%d %s" % (n, word(k)) for k, n in got.items())
    ships0 = {k for k, it in p0.fjord.items() if it["kind"] == "ship"}
    for k, it in p1.fjord.items():
        if it["kind"] == "ship" and k not in ships0:
            text += "; %s (%s +%d)" % (W["ship"], word(it["cat"]), it["bonus"])
    if p1.trophy is not None and p0.trophy is None:
        text += "; %s %d %s" % (W["trophy"], L.TROPHIES[p1.trophy][1], W["vp"])
    fjord = [L.key(c) for c, it in p1.fjord.items() if p0.fjord.get(c) != it
             and it["kind"] in ("res", "bld", "ship")]
    return {"text": text, "cells": cells, "fjord": fjord}


def evaluate(net, games, me):
    """(margin, win) for `me` in each game: exact when the game is over."""
    m, w = np.zeros(len(games)), np.zeros(len(games))
    todo = []
    for k, g in enumerate(games):
        if g.game_over:
            m[k], w[k] = net.E.outcome(g, me)
        else:
            todo.append(k)
    if todo:
        enc = [net.E.encode(games[k], me) for k in todo]
        mm, ww = net(np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]),
                     np.stack([e[2] for e in enc]))
        m[todo], w[todo] = mm, ww
    return m, w


def shares(net, g):
    """Each player's chance to win as the network sees it, adding up to 1
    (the network is asked from every seat)."""
    n = len(g.players)
    if g.game_over:
        w = g.winners()
        return [round(1.0 / len(w), 4) if i in w else 0.0 for i in range(n)]
    ws = np.array([evaluate(net, [g], s)[1][0] for s in range(n)])
    ws = np.clip(ws, 1e-4, None)
    return [round(float(x), 4) for x in ws / ws.sum()]


def classify(loss, gain, gap):
    if gain >= BRILLIANT:
        return "brilliant"
    if loss <= CLASSES[0][1] and gap >= GREAT_GAP:
        return "great"
    for name, most in CLASSES:
        if loss <= most:
            return name


def accuracy(loss):
    return round(100.0 * math.exp(-max(0.0, loss) / ACC_SCALE), 1)


def review_turn(net_name, before, after, search=True, lang="en"):
    """Judge the turn from `before` (the game at the start of the turn) to
    `after` (the game once the turn ended). search=False: only the win
    chances (for a turn of the network itself)."""
    net = nn_bot.load_net(net_name)
    g0 = L.Game.load_state(before)
    g1 = L.Game.load_state(after)
    me = g0.current
    out = {"who": me, "turn": g0.turn_no, "shares_before": shares(net, g0),
           "shares_after": shares(net, g1), "played": describe(g0, g1, me, lang)}
    pm, pw = evaluate(net, [g1], me)
    out["played"].update(margin=round(float(pm[0]), 2), win=round(float(pw[0]), 4))
    if not search or g0.game_over:
        return out
    bot = nn_bot.NNBot(random.Random(0), net_name)
    cands, _ = bot._candidates(copy.deepcopy(g0))
    if not cands:
        return out
    bm, bw = evaluate(net, [c[5] for c in cands], me)
    k = int(np.argmax(bm))
    spaces = lambda c: tuple(sorted(o[0] for o in (c[0], c[2]) if o is not None))
    others = [bm[j] for j, c in enumerate(cands) if spaces(c) != spaces(cands[k])]
    gap = float(bm[k] - max(others)) if others else 0.0
    loss = max(0.0, float(bm[k] - pm[0]))
    gain = max(0.0, float(pm[0] - bm[k]))
    out.update(loss=round(loss, 2), gain=round(gain, 2), gap=round(gap, 2),
               cls=classify(loss, gain, gap), acc=accuracy(loss))
    best = describe(g0, cands[k][5], me, lang)
    if best["text"] == out["played"]["text"] and loss > CLASSES[0][1]:
        best["text"] += (" (other spaces on the fjord)" if lang == "en"
                         else " (andere plekken in de fjord)")
    best.update(margin=round(float(bm[k]), 2), win=round(float(bw[k]), 4))
    if loss > CLASSES[0][1]:
        s = snapshot(cands[k][5])
        s.pop("land")
        s["log"] = s["log"][-14:]
        best["state"] = s             # to show the best move on the board
    out["best"] = best
    return out
