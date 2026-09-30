"""
Build history.html (the history page of the dashboard) from
results/history.json (made by tools/history_data.py): every run, every
network, what changed in the network, the matches between networks, and
how much training data it all took.

    python tools/history_data.py
    python tools/make_history.py
"""

import os
import sys

# This script lives in tools/; the files it uses are in the project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json

# The runs in eras: what the network looked like (2-player runs; t4/m4 are 3-4 players)
ERAS = [
    ("Codering v1", ["v1", "cur", "c2"]),
    ("Codering v3", ["c3"]),
    ("Codering v4", ["c4", "c5v", "c5t", "c7a"]),
    ("Zelfspel", ["s1"]),
    ("Voorstellen", ["p1", "s2"]),
]
ORDER = ["v1", "cur", "c2", "c3", "c4", "c5v", "c5t", "c7a", "s1", "p1", "s2", "m4", "t4"]
MULTI = {"m4", "t4"}                    # played at a table of 4 (1 network, 3 greedy)
ARCH_NL = {"value": "gewoon", "three": "driedelig", "attn": "aandacht", "three4": "driedelig (2-4)",
           "value4": "gewoon (2-4)", "attn4": "aandacht (2-4)"}


def era_of(run):
    for name, runs in ERAS:
        if run in runs:
            return name
    return "3-4 spelers" if run in MULTI else ""


def aggregate(h):
    M, R, D = h["models"], h["runs"], h["data"]
    out = {"made": h["made"], "eras": [[e, r] for e, r in ERAS]}

    # every network that played the full game against greedy (the curve)
    pts, pts_mp = [], []
    for run in ORDER:
        r = R.get(run)
        if not r:
            continue
        for s in r["steps"]:
            if s["stage"] != "full" or s.get("win") is None or s["model"] not in M:
                continue
            row = [M[s["model"]]["t"], round(s["win"], 3), s["model"], run,
                   round(s.get("win_ci") or 0, 3), s.get("games"), s.get("kind") or ""]
            (pts_mp if run in MULTI else pts).append(row)
    if "v1" in M and "cur" in R and R["cur"]["extra"].get("v1 vs greedy"):
        v = R["cur"]["extra"]["v1 vs greedy"]
        pts.insert(0, [M["v1"]["t"], round(v["win"], 3), "v1", "v1", 0, v.get("games"), "greedy data"])
    out["points"], out["points_mp"] = sorted(pts), sorted(pts_mp)

    # the final tests: "X vs greedy" of each run's best networks
    finals = []
    for run in ORDER:
        r = R.get(run)
        if not r:
            continue
        for k, v in r["finals"].items():
            if k.endswith(" vs greedy") and "+2" not in k:
                m = k[:-len(" vs greedy")]
                if m in M:
                    finals.append([M[m]["t"], round(v["win"], 3), m, run, round(v.get("win_ci") or 0, 3),
                                   v.get("games"), round(v.get("margin") or 0, 1)])
    if "v1" in M and R.get("cur", {}).get("extra", {}).get("v1 vs greedy"):
        v = R["cur"]["extra"]["v1 vs greedy"]          # the very first network's test
        finals.append([M["v1"]["t"], round(v["win"], 3), "v1", "v1", 0.058, v.get("games"),
                       round(v.get("margin") or 0, 1)])
    out["finals"] = sorted(finals)

    # the runs table
    runs = []
    for run in ORDER:
        ms = [m for m in M.values() if m["run"] == run]
        if not ms and run not in R:
            continue
        r = R.get(run, {})
        best = r.get("best")
        fin = next((v for k, v in r.get("finals", {}).items()
                    if best and k == best + " vs greedy"), None)
        if run == "v1":
            best, fin = "v1", R.get("cur", {}).get("extra", {}).get("v1 vs greedy")
        if run == "p1":
            best = "p1_r1"
        ds = [s for s in D.values() if s["run"] == run]
        archs = sorted({m["arch"] for m in ms})
        encs = sorted({m["encoding"] for m in ms if m["encoding"]})
        info = h["runs_info"].get(run, ["", ""])
        runs.append({"run": run, "title": info[0], "text": info[1], "era": era_of(run),
                     "start": min((m["made"] for m in ms), default=None),
                     "hours": r.get("hours"), "networks": len(ms),
                     "arch": ", ".join(ARCH_NL.get(a, a) for a in archs), "enc": ", ".join(encs),
                     "weights": max((m["weights"] for m in ms), default=None),
                     "games": sum(s["games"] for s in ds), "positions": sum(s["positions"] for s in ds),
                     "gb": round(sum(s["disk_gb"] for s in ds), 2), "mem_gb": round(sum(s["gb"] for s in ds), 1),
                     "best": best, "best_win": round(fin["win"], 3) if fin else None,
                     "best_margin": round(fin["margin"], 1) if fin and fin.get("margin") is not None else None,
                     "multi": run in MULTI})
    out["runs"] = runs

    # the self-play runs: every challenger against its champion
    sp = {}
    for run in ("s1", "s2", "t4"):
        r = R.get(run)
        if not r:
            continue
        sp[run] = [[int(s["model"].split("_r")[-1]), round(s["vs_champion"], 3), s["model"],
                    s.get("champion"), bool(s.get("accepted")), round(s.get("vs_champion_margin") or 0, 1)]
                   for s in r["steps"] if s.get("vs_champion") is not None]
    out["selfplay"] = sp

    # how well each network judges a position: test error over time, and by moment in the game
    out["mae"] = sorted([m["t"], m["test_mae"], name, m["run"]] for name, m in M.items()
                        if m.get("test_mae") is not None and m["run"] not in MULTI
                        and ("_full" in name or m["run"] in ("v1", "s1", "s2", "p1")))
    milestones = ["v1", "c2_full_r5", "c3_full_r8", "c5t_full_r10", "s1_r43", "s2_r19", "s2_r32"]
    out["phases"] = [[name, M[name]["phases"]] for name in milestones if name in M and M[name].get("phases")]

    # the network's versions: inputs and size
    versions = []
    for name in ["v1", "c3_full_r8", "c5v_full_r10", "c5t_full_r10", "c7a_full_a", "p1_r1", "s2_r32", "t4_r8"]:
        if name in M:
            m = M[name]
            versions.append({"model": name, "made": m["made"], "arch": ARCH_NL.get(m["arch"], m["arch"]),
                             "enc": m["encoding"], "land": m["land_f"], "fjord": m["fjord_f"],
                             "glob": m["glob_f"], "weights": m["weights"], "proposals": m["proposals"]})
    out["versions"] = versions
    out["changes"] = h["changes"]
    out["matches"] = h["matches"]
    out["tuning"] = h["tuning"]

    # totals
    out["totals"] = {
        "networks": len(M), "runs": len([r for r in runs if r["networks"]]),
        "games": sum(s["games"] for s in D.values()), "positions": sum(s["positions"] for s in D.values()),
        "disk_gb": round(sum(s["disk_gb"] for s in D.values()), 1),
        "mem_gb": round(sum(s["gb"] for s in D.values()), 1),
        "hours": round(sum(r.get("hours") or 0 for r in R.values()), 1),
        "first": min(m["made"] for m in M.values()), "last": max(m["made"] for m in M.values()),
        "best2": max((f for f in finals if f[3] not in MULTI), key=lambda f: f[0], default=None),
        "bestmp": max((f for f in finals if f[3] in MULTI), key=lambda f: f[0], default=None),
        "first_win": pts[0][1] if pts else None,
    }
    return out


def main():
    h = json.load(open(os.path.join(ROOT, "results", "history.json"), encoding="utf-8"))
    agg = aggregate(h)
    page = open(os.path.join(ROOT, "templates", "history_template.html"), encoding="utf-8").read()
    with open(os.path.join(ROOT, "history.html"), "w", encoding="utf-8") as f:
        f.write(page.replace("/*DATA*/null", json.dumps(agg, separators=(",", ":"))))
    t = agg["totals"]
    print("history.html: %d networks in %d runs, %.1f GB of training data, %.0f hours"
          % (t["networks"], t["runs"], t["disk_gb"], t["hours"]))


if __name__ == "__main__":
    main()
