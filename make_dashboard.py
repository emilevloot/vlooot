"""
Build the dashboards:

  dashboard.html         the network against itself (selfplay_data.json, made
                         by selfplay_record.py): how games unfold, every game
                         clickable, mistakes found by a deeper check, and a
                         replay of each game in the game page;
  dashboard_greedy.html  the network against the greedy player
                         (dashboard_data.json, made by dashboard_data.py).

    python selfplay_record.py --games 100
    python dashboard_data.py --games 300
    python make_dashboard.py
"""

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = ["wood", "sheep", "gold", "axe", "house", "watchtower", "castle"]
CATS = ["castle", "watchtower", "house", "gold", "sheep", "wood"]


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def aggregate(data):
    games = data["games"]
    n = len(games)
    won = [g for g in games if g["nn_won"] == 1.0]
    lost = [g for g in games if g["nn_won"] == 0.0]
    out = {"model": data["model"], "made": data["made"], "n": n}
    fin = {w: {k: st.mean(g["final"][w][k] for g in games)
               for k in CATS + ["sites", "trophy", "penalty", "total", "filled", "unfilled"]}
           for w in ("nn", "greedy")}
    out["tiles"] = {"win": st.mean(g["nn_won"] for g in games),
                    "score_nn": fin["nn"]["total"], "score_greedy": fin["greedy"]["total"],
                    "margin": st.mean(g["final"]["nn"]["total"] - g["final"]["greedy"]["total"]
                                      for g in games)}
    out["final"] = {"cats": CATS + ["sites", "trophy", "penalty"],
                    "nn": [round(fin["nn"][k], 1) for k in CATS + ["sites", "trophy", "penalty"]],
                    "greedy": [round(fin["greedy"][k], 1) for k in CATS + ["sites", "trophy", "penalty"]]}
    out["ships_filled"] = {w: [round(fin[w]["filled"], 2), round(fin[w]["unfilled"], 2)]
                           for w in ("nn", "greedy")}

    # win chance through the game (the network's own prediction, every turn)
    T = max(len(g["turns"]) for g in games)
    track = {}
    for name, gs in (("won", won), ("lost", lost)):
        rows = []
        for t in range(T):
            xs = [g["turns"][t]["win"] for g in gs if t < len(g["turns"])]
            if len(xs) >= 3:
                rows.append([t + 1, round(q(xs, .5), 3), round(q(xs, .25), 3), round(q(xs, .75), 3), len(xs)])
        track[name] = rows
    out["track"] = track
    out["n_won"], out["n_lost"] = len(won), len(lost)

    # examples: a clear win, a comeback, a loss
    def curve(g):
        return [[t + 1, r["win"], r["who"]] for t, r in enumerate(g["turns"])]
    ex = []
    clear = max(won, key=lambda g: min(r["win"] for r in g["turns"][4:]), default=None)
    comeback = min(won, key=lambda g: min(r["win"] for r in g["turns"]), default=None)
    loss = max(lost, key=lambda g: max(r["win"] for r in g["turns"]), default=None)
    for label, g in (("Duidelijke winst", clear), ("Comeback", comeback), ("Verlies", loss)):
        if g:
            ex.append({"label": label, "seed": g["seed"], "points": curve(g),
                       "final": [g["final"]["nn"]["total"], g["final"]["greedy"]["total"]]})
    out["examples"] = ex

    # calibration: predicted win chance vs how often it really won
    bins = [[] for _ in range(10)]
    for g in games:
        for r in g["turns"]:
            bins[min(int(r["win"] * 10), 9)].append((r["win"], g["nn_won"]))
    out["calib"] = [[round(st.mean(p for p, _ in b), 3), round(st.mean(w for _, w in b), 3), len(b)]
                    for b in bins if len(b) >= 20]

    # what each player takes, per own turn number
    takes = {}
    shiprate = {}
    shipcats = {}
    for w in ("nn", "greedy"):
        per = {}
        shp = {}
        cats = dict.fromkeys(CATS, 0)
        for g in games:
            k = 0
            for r in g["turns"]:
                if r["who"] != w:
                    continue
                k += 1
                per.setdefault(k, []).append(r["got"])
                shp.setdefault(k, []).append(1 if r["ship"] else 0)
                for s in r["ship"]:
                    cats[s["cat"]] += 1
        ks = sorted(k for k in per if len(per[k]) >= n * 0.5)
        takes[w] = {"turns": ks,
                    "items": {it: [round(sum(d.get(it, 0) for d in per[k]) / len(per[k]), 3) for k in ks]
                              for it in ITEMS}}
        shiprate[w] = [[k, round(st.mean(shp[k]), 3)] for k in ks]
        shipcats[w] = [round(cats[c] / n, 3) for c in CATS]
    out["takes"] = takes
    out["totals"] = {w: [round(sum(sum(d.get(it, 0) for d in [r["got"] for r in g["turns"] if r["who"] == w])
                                   for g in games) / n, 2) for it in ITEMS] for w in ("nn", "greedy")}
    out["shiprate"] = shiprate
    out["shipcats"] = {"cats": CATS, **shipcats}

    # shields: in which own turn each is used (average)
    shields = {}
    for w in ("nn", "greedy"):
        when = {"extra": [], "occupy": [], "double": []}
        for g in games:
            k = 0
            for r in g["turns"]:
                if r["who"] == w:
                    k += 1
                    for s in r["shields"]:
                        when[s].append(k)
        shields[w] = {s: [round(st.mean(v), 1) if v else None, round(len(v) / n, 2)] for s, v in when.items()}
    out["shields"] = shields

    # the values the network gives each item, per own turn
    vals = {}
    for g in games:
        k = 0
        for r in g["turns"]:
            if r["who"] == "nn":
                k += 1
                if "values" in r:
                    vals.setdefault(k, []).append(r["values"])
    ks = sorted(k for k in vals if len(vals[k]) >= 20)
    out["values"] = {"turns": ks, "items": ITEMS,
                     # (an item has no value when the fjord is full: left out)
                     "grid": [[round(st.mean([v[it] for v in vals[k] if it in v] or [0]), 2)
                               for k in ks] for it in ITEMS]}
    return out


def mistake_kind(played, better):
    """What differs between the played move and the better one (from the
    texts written by selfplay_record.describe)."""
    if better.endswith("(tegels/schip op andere plekken in de fjord)"):
        return "fjord"
    if better.endswith("(op een ander veld)"):
        return "veld"
    def parts(text):
        segs = text.split("; ")
        return {"place": segs[0].split(" → ")[0],
                "ship": [x for x in segs[1:] if x.startswith("schip")],
                "trophy": [x for x in segs[1:] if x.startswith("trofee")]}
    a, b = parts(played), parts(better)
    if ("schild" in a["place"]) != ("schild" in b["place"]) or a["place"].count("+") != b["place"].count("+"):
        return "schild"
    if a["place"] != b["place"]:
        return "veld"
    if a["ship"] != b["ship"]:
        return "schip"
    if a["trophy"] != b["trophy"]:
        return "trofee"
    return "veld"


KINDS = [("veld", "Verkeerd veld op het bord"), ("schip", "Verkeerd (of geen) langschip"),
         ("schild", "Schild wel of niet gebruiken"), ("trofee", "Trofee"),
         ("fjord", "Tegels/schip op verkeerde plek in de fjord")]


def aggregate_self(data):
    games = data["games"]
    n = len(games)
    out = {"model": data["model"], "made": data["made"], "n": n, "kinds": KINDS}
    rows = []
    swings = []
    by_turn = {}
    kinds = dict.fromkeys([k for k, _ in KINDS], 0)
    first_wins = 0.0
    for g in games:
        T = g["turns"]
        w = g["winners"]
        res0 = (1.0 / len(w)) if 0 in w else 0.0
        first_wins += res0
        series = [r["win0"] for r in T] + [res0]
        ms = []
        turns = []
        for k, r in enumerate(T):
            own = sum(1 for q in T[:k + 1] if q["who"] == r["who"])
            row = {"w": r["who"], "x": r["text"], "p": r["win0"], "g": r["margin0"], "o": own}
            m = r.get("mistake")
            if m:
                kind = mistake_kind(r["text"], m["better"])
                kinds[kind] += 1
                row["m"] = {"l": m["loss"], "v": m["level"], "b": m["better"], "k": kind}
                ms.append(m["loss"])
                by_turn.setdefault(own, [0, 0])[m["level"] - 1] += 1
            turns.append(row)
            if k < len(T) - 1:             # (the last turn only settles the score)
                d = series[k + 1] - series[k]
                swings.append((abs(d), g["seed"], k, round(d, 3), r["who"]))
        winner = w[0] if len(w) == 1 else None
        low = min((series[k] if winner == 0 else 1 - series[k]) for k in range(len(T) - 2)) \
            if winner is not None else None
        big = max(range(len(T) - 1), key=lambda k: abs(series[k + 1] - series[k]))
        rows.append({
            "s": g["seed"], "f": [p["total"] for p in g["final"]], "w": w,
            "fin": [{c: p[c] for c in CATS + ["sites", "trophy", "penalty", "filled", "unfilled"]}
                    for p in g["final"]],
            "series": [round(x, 3) for x in series], "m": len(ms),
            "ml": round(max(ms), 2) if ms else 0, "cb": low is not None and low <= 0.25,
            "low": round(low, 3) if low is not None else None,
            "sw": [big + 1, round(series[big + 1] - series[big], 3)], "T": turns})
    swings.sort(reverse=True)
    out["swings"] = [{"s": s, "t": k + 1, "d": d, "w": who,
                      "x": next(r for r in rows if r["s"] == s)["T"][k]["x"]}
                     for _, s, k, d, who in swings[:8]]
    out["games"] = rows
    out["by_turn"] = [[k] + by_turn.get(k, [0, 0]) for k in range(1, 14)
                      if any(r["o"] == k for g in rows for r in g["T"])]
    out["kind_counts"] = [kinds[k] for k, _ in KINDS]
    nm = sum(r["m"] for r in rows)
    out["tiles"] = {
        "first": first_wins / n,
        "score": st.mean(x for r in rows for x in r["f"]),
        "diff": st.mean(abs(r["f"][0] - r["f"][1]) for r in rows),
        "close": sum(1 for r in rows if abs(r["f"][0] - r["f"][1]) <= 5) / n,
        "mistakes": nm / n, "big": sum(1 for r in rows for t in r["T"] if "m" in t and t["m"]["v"] > 1),
        "clean": sum(1 for r in rows if r["m"] == 0) / n,
        "comebacks": sum(1 for r in rows if r["cb"]),
        "loss": st.mean([t["m"]["l"] for r in rows for t in r["T"] if "m" in t] or [0]),
    }
    # win chance of the first player through the game
    L = max(len(r["series"]) for r in rows)
    band = []
    for k in range(L):
        xs = [r["series"][k] for r in rows if k < len(r["series"]) - 1]
        if len(xs) >= 10:
            band.append([k, round(q(xs, .5), 3), round(q(xs, .25), 3), round(q(xs, .75), 3)])
    out["band"] = band
    # what decided games: points per category, winner vs loser
    win_f, lose_f = [], []
    for r in rows:
        if len(r["w"]) == 1:
            win_f.append(r["fin"][r["w"][0]])
            lose_f.append(r["fin"][1 - r["w"][0]])
    cats = CATS + ["sites", "trophy", "penalty"]
    out["cats"] = {"cats": cats, "win": [round(st.mean(f[c] for f in win_f), 1) for c in cats],
                   "lose": [round(st.mean(f[c] for f in lose_f), 1) for c in cats]}
    return out


def main():
    made = []
    sp = os.path.join(HERE, "selfplay_data.json")
    if os.path.exists(sp):
        agg = aggregate_self(json.load(open(sp)))
        page = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read()
        with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as f:
            f.write(page.replace("/*DATA*/null", json.dumps(agg, separators=(",", ":"))))
        made.append("dashboard.html: %d games of %s against itself, %.1f mistakes per game"
                    % (agg["n"], agg["model"], agg["tiles"]["mistakes"]))
    gp = os.path.join(HERE, "dashboard_data.json")
    if os.path.exists(gp):
        agg = aggregate(json.load(open(gp)))
        page = open(os.path.join(HERE, "dashboard_greedy_template.html"), encoding="utf-8").read()
        with open(os.path.join(HERE, "dashboard_greedy.html"), "w", encoding="utf-8") as f:
            f.write(page.replace("/*DATA*/null", json.dumps(agg)))
        made.append("dashboard_greedy.html: %d games of %s against greedy" % (agg["n"], agg["model"]))
    print("\n".join(made) or "no data: run selfplay_record.py and/or dashboard_data.py first")


if __name__ == "__main__":
    main()
