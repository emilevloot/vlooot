"""
The coach: judges a finished turn the way a chess trainer does.

For the turn of player `me` (the game before and after it, as saved by
Game.save_state) the network lists the moves it would consider itself
(NNBot._candidates: every placement, fjord layout, longship and trophy,
shortlisted) and says where each one ends, in points of final margin.
The move that was played is judged the same way. The difference with the
best move is the points lost, and that gives the label:

    brilliant   better than every move the network found itself, or a
                sacrifice: the best move, banking clearly fewer points
                now than the greedy choice, which is clearly worse later
    great       the best move, and the only good one (the next best
                Viking space is clearly worse)
    best        (nearly) the network's own choice
    excellent / good / inaccuracy / mistake / blunder: more points lost

Accuracy per move falls off with the points lost; a player's accuracy is
the average over their moves. It is one turn deep: the network's own
judgement, not the truth - except in a player's last 2 turns: there the
network's judgement is rough, so the coach counts instead (endgame.py): the
best final score the player could still reach before the move, against the
best it can still reach after it ("exact").

And it says why: what the better move gives that the played one doesn't
(longships filled, sites completed, a trophy, points now, items the network
values for this player, a longship that will stay empty, a shield left
unused), from facts() of both moves - or, for a good move, why it beats the
next best one.

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
CLASSES = [("best", 0.5), ("excellent", 1.2), ("good", 2.0),
           ("inaccuracy", 4.0), ("mistake", 8.0), ("blunder", float("inf"))]
BRILLIANT = 1.0      # points better than the network's own best move
SACRIFICE = 8.0      # or: the best move, banking this many points less now than the
SACRIFICE_WORSE = 5.0  # greedy choice, which is this many points worse in the end
GREAT_GAP = 2.5      # the best move, and the next best space this much worse: the only good move
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


def classify(loss, gain, gap, sacrifice=False):
    if gain >= BRILLIANT or (sacrifice and loss <= CLASSES[0][1]):
        return "brilliant"
    if loss <= CLASSES[0][1] and gap >= GREAT_GAP:
        return "great"
    for name, most in CLASSES:
        if loss <= most:
            return name


SHIELD_WORD = {"extra": "second-Viking", "occupy": "stacking", "double": "double"}


def word(t):
    return ITEM_WORD["en"].get(t, WORDS["en"].get(t, t))


def plural(t):
    """castles, houses - but sheep, gold, wood."""
    return word(t) + ("s" if t in ("castle", "watchtower", "house", "axe") else "")


def listing(items):
    w = [word(m) for m in items]
    return w[0] if len(w) == 1 else ", ".join(w[:-1]) + " and " + w[-1]


def banked(g, me):
    """A player's points so far, without the -5 of longships still empty."""
    sc = g.players[me].score()
    return sc["total"] - sc["penalty"]


def facts(g0, g1, me):
    """What a move gives player `me` (g0: the game before it, g1: after the
    turn): the facts the coach's reasons are made of."""
    p0, p1 = g0.players[me], g1.players[me]
    s0, s1 = p0.score(), p1.score()
    c0, c1 = counts(p0), counts(p1)
    banked = lambda sc: sc["total"] - sc["penalty"]
    f = {"points": banked(s1) - banked(s0),
         "rows": {c: s1["rows"][c]["points"] - s0["rows"][c]["points"] for c in L.SCORE_ORDER},
         "count": {c: s1["rows"][c]["count"] for c in L.SCORE_ORDER},
         "got": {k: c1[k] - c0[k] for k in ITEMS if c1[k] != c0[k]},
         "sites": [it["type"] for c, it in p1.fjord.items()
                   if it["kind"] == "site" and it["done"] and not (p0.fjord.get(c) or {}).get("done")],
         "filled": [(it["cat"], it["bonus"]) for c, it in p1.fjord.items()
                    if it["kind"] == "ship" and it["filled"] and not (p0.fjord.get(c) or {}).get("filled")],
         "trophy": s1["trophy"] - s0["trophy"],
         "turns_left": p1.vikings_left,
         "shields_used": [k for k in ("extra", "occupy", "double") if p0.shields[k] and not p1.shields[k]],
         "shields_left": [k for k in ("extra", "occupy", "double") if p1.shields[k]],
         "empty_ship": None, "new_ship": None}
    for c, it in p1.fjord.items():                     # a new longship, still empty
        if it["kind"] == "ship" and c not in p0.fjord and not it["filled"]:
            missing = list(it["need"])
            for h in p1.adjacent_items(c):
                if h in missing:
                    missing.remove(h)
            room = sum(1 for n in L.neighbours(c) if n in p1.fjord_cells and n not in p1.fjord)
            f["empty_ship"] = {"cat": it["cat"], "missing": missing, "room": room}
        if it["kind"] == "ship" and c not in p0.fjord:
            f["new_ship"] = {"cat": it["cat"], "bonus": it["bonus"]}
    return f


def shopping(p):
    """item -> what of this player's still needs it (an unfilled longship,
    an unfinished construction site)."""
    out = {}
    for c, it in p.fjord.items():
        if (it["kind"] == "ship" and not it["filled"]) or (it["kind"] == "site" and not it["done"]):
            missing = list(it["need"])
            for h in p.adjacent_items(c):
                if h in missing:
                    missing.remove(h)
            name = ("your %s longship" % word(it["cat"]) if it["kind"] == "ship"
                    else "your %s" % L.SITE_NAMES[it["type"]])
            for m in missing:
                out.setdefault(m, name)
    return out


def explain(fa, fb, ma, mb, values=None, needs=None, a_played=True):
    """Why move b is better than move a, in up to 3 reasons, the biggest
    first. fa, fb: their facts(); ma, mb: what each is worth in the end (the
    network's final margin, or the exact final score); values: what one more
    item of each kind is worth to the player now (nn_bot.thoughts); needs:
    shopping() before the move. a_played: a is the move that was played
    (why the best move is better); else b is (why the played move is good)."""
    A = "Your move" if a_played else "The next best move"
    R = []                                              # (points, text)
    for cat, bonus in fb["filled"]:
        if (cat, bonus) not in fa["filled"]:
            n = fb["count"].get(cat, 0)
            R.append((bonus * n + 5.0, "It fills your %s longship: every %s %s %d more%s, and it no longer risks the -5 of an empty longship."
                      % (word(cat), word(cat), "scores" if n else "you collect scores", bonus,
                         " (%+d points now)" % (bonus * n) if n else "")))
    for site in fb["sites"]:
        if site not in fa["sites"]:
            R.append((float(L.SITE_VP[site]), "It completes your %s (+%d points)." % (L.SITE_NAMES[site], L.SITE_VP[site])))
    if fb["trophy"] > fa["trophy"]:
        R.append((float(fb["trophy"] - fa["trophy"]), "It claims the %d-point trophy before your opponent can." % fb["trophy"]))
    es = fa["empty_ship"]
    if es and es["missing"] and not fb["empty_ship"]:
        left = fa["turns_left"]
        hard = len(es["missing"]) > es["room"] or len(es["missing"]) > 2 * left
        R.append((5.0 if hard else 2.5, "%s takes a longship that still needs %s%s: an empty longship costs 5 at the end."
                  % (A, listing(es["missing"]),
                     " - with %d turn%s left that is unlikely" % (left, "" if left == 1 else "s") if hard else "")))
    nb = fb["new_ship"]
    if nb and (not fa["new_ship"] or fa["new_ship"]["cat"] != nb["cat"]) and \
            (nb["cat"], nb["bonus"]) not in fb["filled"]:
        es_b = fb["empty_ship"]
        if es_b and es_b["missing"] and len(es_b["missing"]) <= min(es_b["room"], 2 * fb["turns_left"]):
            n = fb["count"].get(nb["cat"], 0)
            R.append((nb["bonus"] * max(n, 1) * 0.7, "It takes the %s longship: surround it with %s, and every %s scores %d more%s."
                      % (word(nb["cat"]), listing(es_b["missing"]), word(nb["cat"]), nb["bonus"],
                         " (with your %d that is %+d)" % (n, nb["bonus"] * n) if n >= 2 else "")))
    spent = [k for k in fa["shields_used"] if k not in fb["shields_used"]]
    if spent and fa["turns_left"] >= 2:
        R.append((2.0, "%s spends the %s shield here; it is once per game, and worth more on a bigger chance later." % (A, SHIELD_WORD[spent[0]])))
    if fa["turns_left"] == 0 and fa["shields_left"]:
        used = [k for k in fb["shields_used"] if k in fa["shields_left"]]
        if used:
            R.append((3.0, "It uses the %s shield: after the last turn an unused shield is worth nothing." % SHIELD_WORD[used[0]]))
    if values:
        more_b = {k: n - fa["got"].get(k, 0) for k, n in fb["got"].items() if n > fa["got"].get(k, 0)}
        more_a = {k: n - fb["got"].get(k, 0) for k, n in fa["got"].items() if n > fb["got"].get(k, 0)}
        vb = sorted(((values.get(k, 0.0) * n, k) for k, n in more_b.items()), reverse=True)
        va = sorted((values.get(k, 0.0) * n, k) for k, n in more_a.items())
        if vb and vb[0][0] >= 1.0:
            v, k = vb[0]
            why = " (%s needs it)" % needs[k] if needs and k in needs else ""
            instead = (" (%s: %s, %+.1f)" % (A.lower(), word(va[0][1]), va[0][0])) if va and va[0][0] < v - 1.0 else ""
            R.append((v - (va[0][0] if instead else 0.0), "It gets %s%s: the network rates that at %+.1f for you right now%s."
                      % (word(k), why, v, instead)))
    d = fb["points"] - fa["points"]
    if d >= 2 and not any(t.startswith(("It fills", "It completes", "It claims")) for _, t in R):
        parts = sorted(((fb["rows"][c] - fa["rows"][c], c) for c in L.SCORE_ORDER), key=lambda x: -abs(x[0]))
        parts = ", ".join("%s %+d" % (plural(c), p) for p, c in parts[:3] if abs(p) >= 2)
        R.append((float(d), "It scores %d more points right now%s." % (d, " (%s)" % parts if parts else "")))
    later = (mb - ma) - d
    if later >= 2.5 and len(R) < 2:
        R.append((later, "It leaves you better placed for the turns to come (about %.0f points more in the end, by the network)." % later))
    R.sort(key=lambda r: -r[0])
    return [t for _, t in R[:3]]


def accuracy(loss):
    return round(100.0 * math.exp(-max(0.0, loss) / ACC_SCALE), 1)


def review_turn(net_name, before, after, search=True, lang="en", endgame_width="wide"):
    """Judge the turn from `before` (the game at the start of the turn) to
    `after` (the game once the turn ended). search=False: only the win
    chances (for a turn of the network itself). endgame_width: how widely
    the last 2 turns are searched ("wide", "narrow"; None: not counted)."""
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
    if endgame_width and 1 <= g0.players[me].vikings_left <= 2:
        import endgame
        c = endgame.check(g0, g1, me, net_name,
                          endgame.WIDE if endgame_width == "wide" else endgame.NARROW)
        if c is not None:
            loss = c["loss"]
            out.update(loss=round(loss, 2), gain=0.0, gap=0.0, cls=classify(loss, 0.0, 0.0),
                       acc=accuracy(loss), exact=True,
                       reach={"best": c["best"], "played": c["played"],
                              "turns_left": g0.players[me].vikings_left})
            best = describe(g0, c["best_move"][5], me, lang)
            if best["text"] == out["played"]["text"] and loss > CLASSES[0][1]:
                best["text"] += (" (other spaces on the fjord)" if lang == "en"
                                 else " (andere plekken in de fjord)")
            if loss > CLASSES[0][1]:
                s = snapshot(c["best_move"][5])
                s.pop("land")
                s["log"] = s["log"][-14:]
                best["state"] = s
            out["best"] = best
            if loss > CLASSES[0][1]:
                values = nn_bot.thoughts(net, g0, me, only_values=True)["values"]
                out["why"] = explain(facts(g0, g1, me), facts(g0, c["best_move"][5], me),
                                     c["played"], c["best"], values, shopping(g0.players[me]))
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
    # a sacrifice: fewer points now than the greedy choice, which is clearly worse in the end
    now0 = banked(g0, me)
    pts = [banked(c[5], me) - now0 for c in cands]
    played_now = banked(g1, me) - now0
    gi = int(np.argmax(pts))
    sacrifice = bool(pts[gi] - played_now >= SACRIFICE and bm[gi] <= pm[0] - SACRIFICE_WORSE)
    out["greedy"] = {"now": round(float(pts[gi] - played_now), 1), "worse": round(float(pm[0] - bm[gi]), 2)}
    out.update(loss=round(loss, 2), gain=round(gain, 2), gap=round(gap, 2),
               cls=classify(loss, gain, gap, sacrifice), acc=accuracy(loss))
    values = nn_bot.thoughts(net, g0, me, only_values=True)["values"]
    needs = shopping(g0.players[me])
    f_played = facts(g0, g1, me)
    if loss > CLASSES[0][1]:                          # why the best move is better
        out["why"] = explain(f_played, facts(g0, cands[k][5], me), float(pm[0]), float(bm[k]), values, needs)
    else:                                             # why this move is good: against the next best space
        alts = [j for j, c in enumerate(cands) if spaces(c) != spaces(cands[k])]
        if alts:
            j = max(alts, key=lambda j: bm[j])
            good = explain(facts(g0, cands[j][5], me), f_played, float(bm[j]), float(pm[0]), values, needs,
                           a_played=False)
            if sacrifice:
                good.insert(0, "You banked %d points less now than the greedy choice (%s), but that one is about %.0f points worse in the end."
                            % (pts[gi] - played_now, describe(g0, cands[gi][5], me, lang)["text"], pm[0] - bm[gi]))
            out["good"] = good[:3]
            out["runner_up"] = {"text": describe(g0, cands[j][5], me, lang)["text"],
                                "loss": round(float(pm[0] - bm[j]), 2)}
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
