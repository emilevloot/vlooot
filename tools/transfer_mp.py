"""
Turn a trained 2-player network into a 2-4-player network of the same kind
that starts with everything it learned (nn_model.transfer_to_mp).

    python gen_data.py --players 4 --games 500 --name m4_stats
    python tools/transfer_mp.py --from c5t_full_r10 --data m4_stats --name m4_start

--data: multi-player games to measure the scale of the NEW features on.
"""

import os
import sys

# This script lives in tools/; the modules and files it uses are in the
# project folder above it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import glob as globmod

import numpy as np
import torch

import nn_model
import train_nn

HERE = ROOT


def main():
    ap = argparse.ArgumentParser(description="2-player network -> 2-4-player network.")
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--data", required=True, help="multi-player data prefix (for the input scaling)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    src = nn_model.load(os.path.join(train_nn.MODEL_DIR, a.src + ".pt"))
    f = sorted(globmod.glob(os.path.join(train_nn.DATA_DIR, a.data + "_*.npz")))[0]
    d = np.load(f)
    stats = {}
    for name, axes in (("land", (0, 1)), ("glob", (0,))):
        mu, sd = train_nn.feature_stats(d[name], axes)
        stats[name + "_mu"], stats[name + "_sd"] = mu, sd
    dst = nn_model.transfer_to_mp(src, stats)
    torch.save({"state": dst.state_dict(), "arch": dst.ARCH},
               os.path.join(train_nn.MODEL_DIR, a.name + ".pt"))
    np.savez(os.path.join(train_nn.MODEL_DIR, a.name + ".npz"),
             **{k: v.numpy() for k, v in dst.state_dict().items()}, arch=np.array(dst.ARCH))
    print("models/%s: a %s network (%d weights) starting from %s"
          % (a.name, dst.ARCH, nn_model.count_weights(dst), a.src))


if __name__ == "__main__":
    main()
