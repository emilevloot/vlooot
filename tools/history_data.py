"""
Everything the training has made so far, summed up in one small file
(results/history.json) for the history page (history.html, made by
tools/make_history.py): every run, every network, the training data, the
matches between networks and what changed in the network over time.

It keeps the story when the old networks (models/) and the training data
(data/) are cleaned up: run it before deleting them.

    python tools/history_data.py
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import glob
import json
import re
import time
import zipfile

import numpy as np

RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")

# What each run was (written by hand: the files only have the numbers).
RUNS = {
    "v1": ("Het eerste netwerk", "Een waardenetwerk (hex-convoluties over het land en de fjorden) dat leert "
           "van partijen van de greedy-speler: wie wint er, met hoeveel punten?"),
    "cur": ("Eerste curriculum", "De regels stap voor stap (grondstoffen, gebouwen, bouwplaatsen, schepen, "
            "het hele spel), met zelfspel en TD-leren in elke stap."),
    "c2": ("Curriculum 2", "Het curriculum opnieuw, langer; de laatste run met de eerste codering."),
    "c3": ("Codering v3", "Per landveld wat een Viking daar oplevert (tegels, ketens) en een "
           "'boodschappenlijst': wat je schepen en bouwplaatsen nog missen. 80 zelfspelrondes."),
    "c4": ("Run 4 (vastgelopen)", "Gestopt na twee stappen: het GPU-geheugen liep vol (data en netwerk samen). "
           "Daarna: data in het gewone geheugen, per batch naar de GPU."),
    "c5v": ("Codering v4, gewoon netwerk", "Een beurtklok (hoeveel beurten nog) en 'pooling' over het hele "
            "bord, zodat het netwerk verder vooruit kan plannen."),
    "c5t": ("Het driedelige netwerk", "Jouw idee: een fjord-deel (waarde van elk item), een bord-deel (waar "
            "moet ik zetten) en een schip-deel (welk schip past), die elkaar berichten sturen."),
    "c7a": ("Aandacht (transformer)", "Een netwerk met aandacht over alle velden, fjorden en schepen tegelijk. "
            "Te weinig tijd gehad om eerlijk te vergelijken; niet verder gegaan."),
    "s1": ("Zelfspel met een kampioen", "Alleen nog tegen zichzelf: een nieuw netwerk moet de kampioen "
           "verslaan (meer dan 53% van 400 partijen) om hem te vervangen. Vanaf c5t_full_r10."),
    "p1": ("Voorstellen (policy)", "Het netwerk leert ook welke zetten de zoektocht uiteindelijk kiest "
           "(eigen 'torens', zodat de waarde niet slechter wordt), en stelt die zetten zelf voor."),
    "s2": ("Zelfspel met voorstellen", "De voorstellen kiezen welke velden en fjordplekken de zoektocht "
           "probeert; de helft van de partijen op willekeurige borden. Laatste sessie: lagere "
           "leersnelheid, 2 epochs, extra gewicht waar het netwerk ernaast zat, 600 keuringspartijen."),
    "m4": ("Eerste poging 4 spelers", "Het 2-spelernetwerk overgezet naar 4 spelers en getraind op "
           "greedy-partijen: mislukt (4,8% tegen 3 greedy-spelers)."),
    "t4": ("3 en 4 spelers", "s2_r19 overgezet naar de codering voor 2-4 spelers (t4_start: precies "
           "hetzelfde voor 2 spelers), daarna zelfspel met 3 en 4 spelers. Laatste sessie: 3000 "
           "partijen per ronde en een geheugenbudget."),
}

# What changed in the network and around it, in order (the first network of each).
CHANGES = [
    ("v1", "Waardenetwerk", "Hex-convoluties over het landbord en beide fjorden, plus wat "
     "algemene getallen; voorspelt het puntenverschil aan het eind en de winkans."),
    ("cur_resources_a", "Curriculum en TD-leren", "Eerst simpele regels, dan steeds meer; het "
     "netwerk leert ook van zijn eigen oordeel een paar beurten later (minder ruis dan alleen de uitslag)."),
    ("c3_resources_a", "Codering v3", "Per veld: wat een Viking daar zou opleveren (tegels, ketens van "
     "Vikingen) en wat op je 'boodschappenlijst' staat."),
    ("c5v_resources_a", "Codering v4", "Een beurtklok (12 beurten) en pooling-blokken: het hele bord in "
     "een paar getallen, zodat ook verre velden meetellen."),
    ("c5t_resources_a", "Driedelig netwerk", "Fjord-deel, bord-deel en schip-deel met berichten "
     "tussen de delen; de uitleg 'wat het netwerk denkt' in het spel komt hiervandaan."),
    ("c7a_resources_a", "Aandacht", "Een kleine transformer (2 lagen, 2 koppen) over alle velden."),
    ("s1_r1", "Zelfspel met kampioen", "Geen greedy meer als maatstaf: alleen een nieuw netwerk dat de "
     "kampioen verslaat, mag verder. Het netwerk kiest ook zelf de fjordindeling."),
    ("p1_r1", "Voorstellen", "Extra uitgangen (eigen torens) die voorspellen welke plaatsing, "
     "fjordplek en welk schip de zoektocht kiest; ze stellen zelf kandidaten voor."),
    ("s2_r1", "Voorstellen in de zoektocht", "De voorstellen kiezen welke velden en fjordplekken "
     "geprobeerd worden (plus 1 greedy-plek als vangnet); later ook willekeurige borden."),
    ("t4_start", "2-4 spelers", "Een codering voor 2-4 spelers (alle 4 borden, 'de tegenstander' = de "
     "sterkste); een 2-spelernetwerk wordt precies overgezet."),
    ("s2_r24", "Leren waar het mis ging", "Lagere leersnelheid en 2 epochs (het netwerk leerde de data "
     "uit het hoofd), posities waar het ernaast zat tellen zwaarder, en de laatste 2 beurten worden "
     "exact uitgerekend in plaats van geschat."),
]


# Matches measured by hand (not in a results file)
EXTRA_MATCHES = [
    ("s2_r19!e", "s2_r19", 0.535, 0.049, 0.77, 400,
     "het eindspel exact rekenen (endgame.py) tegen alleen het netwerk"),
    ("s2_r32", "s2_r19", 0.650, 0.047, 6.37, 400, "tussentijdse meting tijdens de 8-uurstraining"),
]


def npz_headers(path):
    """{array name: (shape, dtype)} of a .npz file, without unpacking the arrays."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                   else np.lib.format.read_array_header_2_0(f))
            out[name[:-4] if name.endswith(".npy") else name] = (shape, dtype)
    return out


def run_of(name):
    return name.split("_")[0]


def date(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def encoding_of(land_f):
    return {19: "v1", 36: "v3", 38: "v4", 39: "v4 (2-4 spelers)"}.get(land_f, "%d kenmerken" % land_f)


def models():
    out = {}
    for path in sorted(glob.glob(os.path.join(MODELS, "*.npz"))):
        name = os.path.basename(path)[:-4]
        try:
            h = npz_headers(path)
        except (OSError, zipfile.BadZipFile, ValueError):
            continue
        land_f = h["land.c1.w"][0][1] if "land.c1.w" in h else (h["land_mu"][0][0] if "land_mu" in h else None)
        fjord_f = h["fjord.c1.w"][0][1] if "fjord.c1.w" in h else None
        glob_f = h["glob.weight"][0][1] if "glob.weight" in h else None
        weights = sum(int(np.prod(s)) for k, (s, d) in h.items()
                      if d.kind == "f" and not k.endswith(("_mu", "_sd")))
        arch = "value"
        if "arch" in h:
            with np.load(path) as z:
                arch = str(z["arch"])
        out[name] = {"run": run_of(name), "made": date(os.path.getmtime(path)),
                     "t": os.path.getmtime(path), "arch": arch, "land_f": land_f,
                     "fjord_f": fjord_f, "glob_f": glob_f, "encoding": encoding_of(land_f) if land_f else None,
                     "weights": weights, "proposals": any(k.startswith("pol_") for k in h),
                     "kb": round(os.path.getsize(path) / 1024)}
        rep = os.path.join(MODELS, name + "_train.json")
        if os.path.exists(rep):
            try:
                r = json.load(open(rep))
            except ValueError:
                r = {}
            eps = r.get("epochs", [])
            if eps:
                out[name]["test_mae"] = round(min(e["test_mae"] for e in eps), 2)
                out[name]["epochs"] = len(eps)
                out[name]["best_epoch"] = r.get("best_epoch")
                out[name]["train_s"] = round(eps[-1].get("time_s", 0))
            if r.get("phases"):
                out[name]["phases"] = [[p["vikings_left"], round(p["network"], 2), round(p["score"], 2)]
                                       for p in r["phases"]]
            if "baseline_score" in r:
                out[name]["baseline"] = round(r["baseline_score"], 2)
    return out


def data_sets():
    """Positions, games and memory per data set (from the file headers; the
    game numbers are read from the small 'game' arrays)."""
    sets = {}
    for path in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
        base = os.path.basename(path)[:-4]
        name = re.sub(r"_\d{3}$", "", base)
        try:
            h = npz_headers(path)
            rows = int(h["margin"][0][0]) if "margin" in h else 0
            nbytes = sum(int(np.prod(s)) * d.itemsize for s, d in h.values())
            games = 0
            if "game" in h:
                with np.load(path) as z:
                    games = len(np.unique(z["game"]))
        except (OSError, zipfile.BadZipFile, ValueError, KeyError):
            continue
        s = sets.setdefault(name, {"run": run_of(name), "files": 0, "positions": 0, "games": 0,
                                   "gb": 0.0, "disk_gb": 0.0, "t": os.path.getmtime(path)})
        s["files"] += 1
        s["positions"] += rows
        s["games"] += games
        s["gb"] += nbytes / 1e9
        s["disk_gb"] += os.path.getsize(path) / 1e9
        s["t"] = min(s["t"], os.path.getmtime(path))
    for s in sets.values():
        s["gb"] = round(s["gb"], 3)
        s["disk_gb"] = round(s["disk_gb"], 3)
        s["made"] = date(s["t"])
    return sets


def log_hours(prefix):
    """Hours of work in a run's log: the time between its lines, leaving out
    pauses of more than 90 minutes (between sessions)."""
    path = os.path.join(RESULTS, "curriculum_%s_log.txt" % prefix)
    if not os.path.exists(path):
        return None
    ts = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"(\d\d):(\d\d):(\d\d) ", line)
        if m:
            ts.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
    total = 0
    for a, b in zip(ts, ts[1:]):
        d = b - a if b >= a else b + 86400 - a
        if d <= 90 * 60:
            total += d
    return round(total / 3600, 2)


def runs():
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "curriculum_*_results.json"))):
        prefix = os.path.basename(path)[len("curriculum_"):-len("_results.json")]
        r = json.load(open(path))
        steps = []
        for stage, st in r.get("stages", {}).items():
            for s in st.get("steps", []):
                a = s.get("arena") or {}
                row = {"model": s["model"], "stage": stage, "kind": s.get("kind"),
                       "test_mae": s.get("test_mae"), "win": a.get("win"),
                       "win_ci": a.get("win_ci"), "margin": a.get("margin"), "score": a.get("score"),
                       "games": a.get("games"), "ms": a.get("ms_per_turn")}
                vc = s.get("vs_champion")
                if vc:
                    row.update(vs_champion=vc["win"], vs_champion_margin=vc.get("margin"),
                               champion=s.get("champion"), accepted=s.get("accepted"))
                steps.append(row)
        finals = {k: {"win": v.get("win"), "win_ci": v.get("win_ci"), "margin": v.get("margin"),
                      "games": v.get("games")}
                  for k, v in r.get("final", {}).items() if isinstance(v, dict) and "win" in v}
        best = r.get("stages", {}).get("full", {}).get("best") or r.get("stages", {}).get("full", {}).get("champion")
        extra = {}
        if "v1_full_game" in r:
            extra["v1 vs greedy"] = r["v1_full_game"]
        out[prefix] = {"steps": steps, "finals": finals, "best": best, "extra": extra,
                       "hours": log_hours(prefix), "selfplay": r.get("selfplay")}
    return out


def matches(run_results):
    """Networks against each other: the compare_*.json matches, and every
    final 'X vs Y (where it started)'."""
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "compare_*.json"))):
        m = json.load(open(path))
        out.append({"a": m["a"], "b": m["b"], "win": m["a_win"], "ci": m.get("a_win_ci"),
                    "margin": m.get("a_margin"), "games": m["games"]})
    for prefix, r in run_results.items():
        for k, v in r["finals"].items():
            m = re.match(r"(\S+) vs (\S+) \((.*)\)$", k)
            if m and "greedy" not in m.group(2):
                out.append({"a": m.group(1), "b": m.group(2), "win": v["win"], "ci": v.get("win_ci"),
                            "margin": v.get("margin"), "games": v.get("games"), "note": m.group(3)})
    for a, b, win, ci, margin, games, note in EXTRA_MATCHES:
        out.append({"a": a, "b": b, "win": win, "ci": ci, "margin": margin, "games": games, "note": note})
    return out


def tuning():
    path = os.path.join(RESULTS, "tune_history.json")
    if not os.path.exists(path):
        return None
    h = json.load(open(path))
    pts = [[r.get("round", i + 1), round(r["champ_score"], 2)]
           for i, r in enumerate(h.get("rounds", [])) if "champ_score" in r]
    fin = h.get("final") or {}
    return {"points": pts, "final_adv": fin.get("adv"), "final_score": fin.get("score"),
            "original_score": fin.get("original_score"), "games": fin.get("games")}


def main():
    t0 = time.time()
    M = models()
    R = runs()
    D = data_sets()
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "runs_info": RUNS,
           "changes": [{"model": m, "title": t, "text": x,
                        "made": M[m]["made"] if m in M else None} for m, t, x in CHANGES],
           "models": M, "runs": R, "data": D, "matches": matches(R), "tuning": tuning()}
    with open(os.path.join(RESULTS, "history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print("results/history.json: %d networks, %d runs, %d data sets (%.1f GB on disk, %d positions), "
          "%d matches, in %.0f s"
          % (len(M), len(R), len(D), sum(s["disk_gb"] for s in D.values()),
             sum(s["positions"] for s in D.values()), len(out["matches"]), time.time() - t0))


if __name__ == "__main__":
    main()
