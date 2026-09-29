"""
The value network in PyTorch on the GPU, for self-play (gen_data.py): the
same numbers as nn_bot's numpy networks (up to rounding), but fast for
batches of thousands of positions. Works for every kind of network in
nn_model.ARCHS.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class TorchNet:
    CHUNK = 4096            # positions per step on the GPU (limits memory)

    def __init__(self, name, device=None):
        import torch
        import nn_model
        path = name if name.endswith(".pt") else os.path.join(HERE, "models", name + ".pt")
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn_model.load(path).eval().to(self.dev)
        self.has_proposals = hasattr(self.model, "pol_land")
        if self.model.ARCH == "attn" or hasattr(self.model.E, "PRESENT_COLS"):
            # attention compares every token with every other, and a 2-4-player
            # position is twice as big: fewer positions per step, or 6
            # processes together fill the GPU (and everything crawls)
            self.CHUNK = 1024

    @staticmethod
    def _unique(x):
        """Rows of x without repeats of the row just before, and for every row
        which of those it is. The candidate turns come in groups (all endings
        of one Viking placement, all turns of one game) whose boards repeat
        in a row, so comparing each row with the previous one finds almost
        every duplicate - with one quick comparison instead of sorting."""
        flat = x.reshape(len(x), -1)
        new = np.empty(len(x), np.bool_)
        new[0] = True
        new[1:] = (flat[1:] != flat[:-1]).any(1)
        return x[new], np.cumsum(new) - 1

    def _board(self, x, board, mu, sd, mask_col=None):
        """board.parts() once per DIFFERENT board, put back in place."""
        torch = self.torch
        u, inv = self._unique(x)
        raw = torch.from_numpy(u).to(self.dev)
        mask = None if mask_col is None else raw[:, :, mask_col:mask_col + 1].float()
        a, cells = board.parts((raw.float() - mu) / sd, mask)
        inv = torch.from_numpy(inv).to(self.dev)
        return a[inv], cells[inv]

    def __call__(self, land, fjord, glob):
        """Many positions share boards: the opponent's fjord is the same in
        all candidate turns of a game, the landscape the same for all
        endings of one Viking placement. Each board part of the network runs
        once per different board; the rest (the "head") for every position."""
        torch = self.torch
        mdl = self.model
        ms, ws = [], []
        with torch.no_grad():
            for i in range(0, len(land), self.CHUNK):
                part = slice(i, i + self.CHUNK)
                a, cells = self._board(land[part], mdl.land, mdl.land_mu, mdl.land_sd,
                                       getattr(mdl.E, "LF_INPLAY", None))
                seats = [self._board(fjord[part, k], mdl.fjord, mdl.fjord_mu, mdl.fjord_sd)
                         for k in range(fjord.shape[1])]
                gl = torch.from_numpy(glob[part]).to(self.dev)
                b, c, fb, fc = mdl.seats(torch.stack([x[0] for x in seats], 1),
                                         torch.stack([x[1] for x in seats], 1), gl)
                m, w, _ = mdl.head(torch.from_numpy(land[part]).to(self.dev), gl,
                                   a, cells, b, c, (fb, fc))
                ms.append(m * 20.0)
                ws.append(torch.sigmoid(w))
        return torch.cat(ms).cpu().numpy(), torch.cat(ws).cpu().numpy()

    def proposals(self, land, fjord, glob):
        """The proposals (nn_model.ThreePartNet.proposals) for positions at the
        start of a turn, as numpy: pol_place, pol_tile, pol_ship."""
        torch = self.torch
        self.model.with_proposals = True
        out = {}
        with torch.no_grad():
            for i in range(0, len(land), self.CHUNK):
                part = slice(i, i + self.CHUNK)
                _, _, told = self.model.explain(*(torch.from_numpy(x[part]).to(self.dev)
                                                  for x in (land, fjord, glob)))
                for k in ("pol_place", "pol_tile", "pol_ship"):
                    out.setdefault(k, []).append(told[k].cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}
