"""
The neural network that judges a Looot position (PyTorch).

It predicts, for the player "me" of the encoding (nn_encode.py):
  margin  my final score minus the opponent's (in units of 20 points)
  win     how likely I am to win (as a logit; sigmoid gives the chance)

Building block: a hex convolution. For every space it combines the space
itself with its 6 neighbours, each direction with its own weights - like
the 3x3 window of an image network, but for hexagons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_encode as E

MARGIN_SCALE = 20.0     # the network predicts margin / 20 (easier numbers)


class HexConv(nn.Module):
    """out[space] = W0 * x[space] + sum over directions d of Wd * x[neighbour d]"""

    def __init__(self, cin, cout, nb):
        super().__init__()
        self.register_buffer("nb", torch.as_tensor(nb, dtype=torch.long))
        self.w = nn.Parameter(torch.randn(7, cin, cout) * (2.0 / (7 * cin)) ** 0.5)
        self.b = nn.Parameter(torch.zeros(cout))

    def forward(self, x):                      # x: [batch, spaces, cin]
        pad = torch.zeros_like(x[:, :1])       # "no neighbour" = zeros
        xp = torch.cat([x, pad], dim=1)
        nbx = xp[:, self.nb]                   # [batch, spaces, 6, cin]
        out = x @ self.w[0] + torch.einsum("bndc,dco->bno", nbx, self.w[1:])
        return out + self.b


class Board(nn.Module):
    """A stack of hex convolutions over one board, squeezed to `out` numbers."""

    def __init__(self, cin, width, squeeze, nb, n_spaces, out):
        super().__init__()
        self.c1 = HexConv(cin, width, nb)
        self.c2 = HexConv(width, width, nb)
        self.c3 = HexConv(width, width, nb)
        self.c4 = HexConv(width, squeeze, nb)
        self.fc = nn.Linear(n_spaces * squeeze, out)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(h + self.c2(h))             # "residual": add, don't replace
        h = F.relu(h + self.c3(h))
        h = F.relu(self.c4(h))
        return F.relu(self.fc(h.flatten(1)))


class ValueNet(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        # dropout: while training, randomly switch off this share of the
        # neurons so the network can't memorise games (off while playing)
        self.drop = nn.Dropout(dropout)
        self.land = Board(E.LAND_F, 48, 16, E.LAND_NB, E.N_LAND, 128)
        self.fjord = Board(E.FJORD_F, 32, 12, E.FJORD_NB, E.N_FJORD, 64)
        self.glob = nn.Linear(E.GLOB_F, 64)
        self.fc1 = nn.Linear(128 + 64 + 64 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.margin = nn.Linear(128, 1)
        self.win = nn.Linear(128, 1)
        # input scaling (mean/std per feature), filled in from the data
        self.register_buffer("land_mu", torch.zeros(E.LAND_F))
        self.register_buffer("land_sd", torch.ones(E.LAND_F))
        self.register_buffer("fjord_mu", torch.zeros(E.FJORD_F))
        self.register_buffer("fjord_sd", torch.ones(E.FJORD_F))
        self.register_buffer("glob_mu", torch.zeros(E.GLOB_F))
        self.register_buffer("glob_sd", torch.ones(E.GLOB_F))

    def forward(self, land, fjord, glob):
        """land [B,55,19], fjord [B,2,37,30], glob [B,104] (raw numbers).
        Returns (margin / 20, win logit), each [B]."""
        land = (land.float() - self.land_mu) / self.land_sd
        fjord = (fjord.float() - self.fjord_mu) / self.fjord_sd
        glob = (glob.float() - self.glob_mu) / self.glob_sd
        a = self.land(land)
        b = self.fjord(fjord[:, 0])            # my fjord
        c = self.fjord(fjord[:, 1])            # opponent's fjord, same layers
        d = F.relu(self.glob(glob))
        h = self.drop(torch.cat([a, b, c, d], dim=1))
        h = self.drop(F.relu(self.fc1(h)))
        h = F.relu(self.fc2(h))
        return self.margin(h).squeeze(1), self.win(h).squeeze(1)


def count_weights(model):
    return sum(p.numel() for p in model.parameters())
