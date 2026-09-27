"""
Train the value network on data made by gen_data.py.

    python train_nn.py                         # data/greedy_*.npz -> models/v1
    python train_nn.py --data greedy selfplay1 --name v2 --epochs 12

What happens:
  1. All positions are loaded and split BY GAME into training games (90%)
     and test games (10%) the network never learns from.
  2. Every input feature is scaled to mean 0 / spread 1, using the
     training data (networks learn much better from numbers of that size).
  3. Training: the network predicts batches of positions, the error is
     measured (the "loss"), and every weight is nudged to shrink it
     (gradient descent, done on the GPU).
  4. After every pass over the data (an "epoch") the test games are
     predicted, and compared with two simple guesses:
       "draw"   - always predict a margin of 0
       "score"  - predict the current score difference as the final margin
  5. The trained network is saved to models/<name>.pt (PyTorch) and
     models/<name>.npz (plain numbers, used by nn_bot.py to play).
"""

import argparse
import glob as globmod
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import nn_encode as E
from nn_model import MARGIN_SCALE, ValueNet, count_weights

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")


def load(prefixes):
    files = sorted(f for p in prefixes
                   for f in globmod.glob(os.path.join(DATA_DIR, p + "_*.npz")))
    if not files:
        raise SystemExit("No data files found for %s in data/" % prefixes)
    # one kind of array at a time, so only that kind is in memory twice
    # while it is glued together (the data can be several GB)
    zips = [np.load(f) for f in files]
    out = {}
    for k in ("land", "fjord", "glob", "margin", "win", "game"):
        out[k] = np.concatenate([z[k] for z in zips])
    print("Loaded %d positions from %d games (%d files)."
          % (len(out["margin"]), len(np.unique(out["game"])), len(files)))
    return out


def feature_stats(x, axes):
    x = x.astype(np.float32)
    mu = x.mean(axis=axes)
    sd = x.std(axis=axes)
    sd[sd < 1e-3] = 1.0          # features that never change: leave as they are
    return mu, sd


def main():
    ap = argparse.ArgumentParser(description="Train the value network.")
    ap.add_argument("--data", nargs="+", default=["greedy"], help="data file prefixes")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096,
                    help="positions per training step (big batches keep the GPU busy)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init", default=None, help="start from this model instead of random")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="share of neurons switched off while training")
    ap.add_argument("--wd", type=float, default=0.01,
                    help="weight decay: penalty on large weights")
    ap.add_argument("--td", type=int, default=0,
                    help="TD learning: also learn from the --init network's own "
                         "judgement N turns later (0 = only the final results)")
    ap.add_argument("--td-mix", type=float, default=0.5,
                    help="with --td: share of the final result in the target")
    ap.add_argument("--stats-from", default=None,
                    help="data prefix to take the input scaling from (e.g. the full "
                         "game's data when training on a simpler rule stage)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")
    D = load(a.data)

    # --- 1. split by game: every 10th game is a test game ---
    test = (D["game"] % 10) == 0
    tr, te = np.where(~test)[0], np.where(test)[0]
    print("Training positions: %d, test positions: %d" % (len(tr), len(te)))

    # --- 2. feature scaling from a sample of the training data ---
    sample = np.random.default_rng(0).choice(tr, size=min(200000, len(tr)), replace=False)
    model = ValueNet(a.dropout)
    if a.init:
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, a.init + ".pt"))["state"])
    else:
        S = {k: D[k][sample] for k in ("land", "fjord", "glob")}
        if a.stats_from:
            f = sorted(globmod.glob(os.path.join(DATA_DIR, a.stats_from + "_*.npz")))[0]
            S = {k: v for k, v in np.load(f).items() if k in ("land", "fjord", "glob")}
            print("Input scaling taken from", os.path.basename(f))
        for name, x, axes in (("land", S["land"], (0, 1)),
                              ("fjord", S["fjord"], (0, 1, 2)),
                              ("glob", S["glob"], (0,))):
            mu, sd = feature_stats(x, axes)
            getattr(model, name + "_mu").copy_(torch.from_numpy(mu))
            getattr(model, name + "_sd").copy_(torch.from_numpy(sd))
    model.to(dev)
    print("Network has %d weights." % count_weights(model))

    # all data on the GPU as small integers; converted per batch
    T = {k: torch.from_numpy(D[k]).to(dev) for k in ("land", "fjord", "glob", "margin", "win")}

    # --- TD learning: a less noisy answer key ---
    # The final result of a game contains all the luck of the turns still to
    # come. Instead, the target becomes a mix of that final result and what
    # the previous network (--init) thought of the SAME player's position N
    # turns later. Positions are stored per game as turn 0 (player 0, player
    # 1), turn 1 (player 0, player 1), ..., so N turns later is 2N rows on.
    if a.td:
        if not a.init:
            raise SystemExit("--td needs --init (the network whose judgement is used)")
        model.eval()
        pm_all, pw_all = [], []
        with torch.no_grad():
            for i in range(0, len(D["margin"]), 16384):
                s = slice(i, i + 16384)
                pm, pw = model(T["land"][s], T["fjord"][s], T["glob"][s])
                pm_all.append(pm * MARGIN_SCALE)
                pw_all.append(torch.sigmoid(pw))
        model.train()
        pm_all, pw_all = torch.cat(pm_all), torch.cat(pw_all)
        step = 2 * a.td
        n = len(D["margin"])
        later = torch.arange(n, device=dev) + step
        game = torch.from_numpy(D["game"]).to(dev)
        ok = later < n
        ok[ok.clone()] &= game[later[ok]] == game[ok]      # same game still?
        # the very last positions of a game are exact: keep the real result there
        mix = a.td_mix
        tm, tw = T["margin"].clone(), T["win"].clone()
        idx = torch.where(ok)[0]
        tm[idx] = mix * T["margin"][idx] + (1 - mix) * pm_all[later[idx]]
        tw[idx] = mix * T["win"][idx] + (1 - mix) * pw_all[later[idx]]
        print("TD targets: %d of %d positions use the network's judgement %d turns later "
              "(mix %.0f%% final result)" % (int(ok.sum()), n, a.td, 100 * mix))
        T["target_margin"], T["target_win"] = tm, tw
    else:
        T["target_margin"], T["target_win"] = T["margin"], T["win"]
    tr_t = torch.from_numpy(tr).to(dev)
    te_t = torch.from_numpy(te).to(dev)

    # --- baselines on the test games ---
    m_te = D["margin"][te]
    score_diff = (D["glob"][te, E.G["total_me"]] - D["glob"][te, E.G["total_opp"]]).astype(np.float32)
    base_draw = float(np.abs(m_te).mean())
    base_score = float(np.abs(m_te - score_diff).mean())
    print("Test error of simple guesses (average points off): "
          "'draw' %.1f, 'current score' %.1f" % (base_draw, base_score))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.1)

    def losses(idx):
        # trained towards the targets (TD or final result); the test error
        # below is always measured against the real final results
        pm, pw = model(T["land"][idx], T["fjord"][idx], T["glob"][idx])
        lm = F.smooth_l1_loss(pm, T["target_margin"][idx] / MARGIN_SCALE)
        lw = F.binary_cross_entropy_with_logits(pw, T["target_win"][idx])
        return lm, lw, pm, pw

    @torch.no_grad()
    def evaluate():
        model.eval()
        preds, wins = [], []
        for i in range(0, len(te_t), 8192):
            _, _, pm, pw = losses(te_t[i:i + 8192])
            preds.append(pm * MARGIN_SCALE)
            wins.append(torch.sigmoid(pw))
        model.train()
        pm = torch.cat(preds).cpu().numpy()
        pw = torch.cat(wins).cpu().numpy()
        mae = float(np.abs(pm - m_te).mean())
        w_true = D["win"][te]
        decided = w_true != 0.5
        acc = float(((pw[decided] > 0.5) == (w_true[decided] > 0.5)).mean())
        return mae, acc, pm

    history = {"baseline_draw": base_draw, "baseline_score": base_score, "epochs": []}
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    best = None
    for ep in range(1, a.epochs + 1):
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        tot_m = tot_w = 0.0
        n = 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            lm, lw, _, _ = losses(idx)
            loss = lm + lw
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_m += lm.item()
            tot_w += lw.item()
            n += 1
        mae, acc, _ = evaluate()
        history["epochs"].append({"epoch": ep, "train_margin_loss": tot_m / n,
                                  "train_win_loss": tot_w / n, "test_mae": mae,
                                  "test_win_acc": acc, "time_s": time.time() - t0})
        mark = ""
        if best is None or mae < best[0]:
            # early stopping: remember the epoch that does best on test games
            best = (mae, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
            mark = "  <- best so far"
        print("epoch %2d: train loss %.3f + %.3f | test: %.2f points off, "
              "winner right %.1f%% | %.0f s%s"
              % (ep, tot_m / n, tot_w / n, mae, 100 * acc, time.time() - t0, mark), flush=True)

    model.load_state_dict(best[2])
    history["best_epoch"] = best[1]
    print("Keeping epoch %d (%.2f points off)." % (best[1], best[0]))

    # --- how good is it at different moments of the game? ---
    mae, acc, pm = evaluate()
    left = D["glob"][te, E.G["vik_me"]] + D["glob"][te, E.G["vik_opp"]]
    phases = []
    print("\nTest error by moment in the game (points off):")
    print("  Vikings left   network   'current score'   'draw'")
    for lo, hi in ((21, 26), (16, 20), (11, 15), (6, 10), (1, 5), (0, 0)):
        sel = (left >= lo) & (left <= hi)
        if sel.any():
            row = {"vikings_left": "%d-%d" % (lo, hi) if lo != hi else str(lo),
                   "network": float(np.abs(pm[sel] - m_te[sel]).mean()),
                   "score": float(np.abs(score_diff[sel] - m_te[sel]).mean()),
                   "draw": float(np.abs(m_te[sel]).mean())}
            phases.append(row)
            print("  %-12s %9.1f %17.1f %8.1f" % (row["vikings_left"], row["network"],
                                                  row["score"], row["draw"]))
    history["phases"] = phases

    # --- save ---
    model.cpu()
    torch.save({"state": model.state_dict()}, os.path.join(MODEL_DIR, a.name + ".pt"))
    np.savez(os.path.join(MODEL_DIR, a.name + ".npz"),
             **{k: v.numpy() for k, v in model.state_dict().items()})
    with open(os.path.join(MODEL_DIR, a.name + "_train.json"), "w") as f:
        json.dump(history, f, indent=1)
    print("\nSaved models/%s.pt and models/%s.npz" % (a.name, a.name))


if __name__ == "__main__":
    main()
