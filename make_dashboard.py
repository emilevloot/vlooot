"""
Build dashboard.html from dashboard_data.json (made by dashboard_data.py):
how the network plays - its win chance through the game, how well those
predictions come true, what it takes each turn compared with the greedy
player, which longships, where its points come from, and the value it gives
each item as the game goes on.

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


PAGE = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read() \
    if os.path.exists(os.path.join(HERE, "dashboard_template.html")) else None


def main():
    data = json.load(open(os.path.join(HERE, "dashboard_data.json")))
    agg = aggregate(data)
    page = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read()
    page = page.replace("/*DATA*/null", json.dumps(agg))
    with open(os.path.join(HERE, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print("dashboard.html: %d games of %s against greedy" % (agg["n"], agg["model"]))


if __name__ == "__main__":
    main()
