"""Ensemble of small autoencoders (the KitNET idea), own PyTorch code.

1. Features are grouped by correlation (features that move together share an autoencoder), each group at most
   `max_group` wide. Small groups mean small networks: cheap, and an anomaly in one corner of the feature space is
   not averaged away by the other 90 features.
2. One autoencoder per group is trained on NORMAL data only; its reconstruction RMSE says how unusual that group is.
3. An output autoencoder is trained on the vector of group RMSEs. Its own reconstruction RMSE is the anomaly score.

Nothing here has seen an attack. Anomaly means "does not look like the normal traffic we trained on".
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from torch import nn


def correlation_groups(X: np.ndarray, max_group: int) -> list[np.ndarray]:
    """Split feature indices into groups of at most `max_group`, putting correlated features together."""
    d = X.shape[1]
    if d <= max_group:
        return [np.arange(d)]
    Xc = X - X.mean(0)
    sd = Xc.std(0)
    ok = sd > 1e-9
    corr = np.zeros((d, d))
    corr[np.ix_(ok, ok)] = np.corrcoef(Xc[:, ok].T)
    dist = 1.0 - np.abs(np.nan_to_num(corr))
    np.fill_diagonal(dist, 0.0)
    tree = to_tree(linkage(squareform(dist, checks=False), method="average"))
    groups: list[np.ndarray] = []

    def walk(node):
        if node.get_count() <= max_group:
            groups.append(np.array(sorted(node.pre_order())))
        else:
            walk(node.get_left())
            walk(node.get_right())

    walk(tree)
    return groups


class _AE(nn.Module):
    def __init__(self, n_in: int, ratio: float):
        super().__init__()
        h = max(1, math.ceil(n_in * ratio))
        self.enc = nn.Linear(n_in, h)
        self.dec = nn.Linear(h, n_in)

    def forward(self, x):
        return torch.sigmoid(self.dec(torch.sigmoid(self.enc(x))))


def _rmse(model: _AE, x: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]
            out.append(torch.sqrt(((model(xb) - xb) ** 2).mean(1)))
    return torch.cat(out)


def _train(model: _AE, x: torch.Tensor, epochs: int, batch: int, lr: float, gen: torch.Generator) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            xb = x[order[i:i + batch]]
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()


class AutoencoderEnsemble:
    def __init__(self, max_group: int = 10, hidden_ratio: float = 0.75, epochs: int = 12, batch: int = 256,
                 lr: float = 5e-3, seed: int = 0):
        self.max_group, self.hidden_ratio, self.epochs, self.batch, self.lr, self.seed = (
            max_group, hidden_ratio, epochs, batch, lr, seed)
        self.groups: list[np.ndarray] = []
        self.nets: list[_AE] = []
        self.out_net: _AE | None = None

    # -- scaling: min-max from normal training data only; values outside the range are NOT clipped, so an
    #    out-of-range feature produces a large reconstruction error (that is the point).
    def _scale(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(((X - self.lo) / self.span).astype(np.float32))

    def _group_rmse(self, Xs: torch.Tensor) -> torch.Tensor:
        cols = [_rmse(net, Xs[:, g]) for net, g in zip(self.nets, self.groups)]
        return torch.stack(cols, dim=1)

    def fit(self, X: np.ndarray) -> "AutoencoderEnsemble":
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        self.lo = X.min(0)
        self.span = np.where(X.max(0) - self.lo > 1e-9, X.max(0) - self.lo, 1.0)
        Xs = self._scale(X)
        self.groups = correlation_groups(X, self.max_group)
        self.nets = []
        for g in self.groups:
            net = _AE(len(g), self.hidden_ratio)
            _train(net, Xs[:, g], self.epochs, self.batch, self.lr, gen)
            net.eval()
            self.nets.append(net)
        R = self._group_rmse(Xs)
        self.r_hi = R.max(0).values.clamp_min(1e-6)
        self.out_net = _AE(R.shape[1], self.hidden_ratio)
        _train(self.out_net, R / self.r_hi, self.epochs, self.batch, self.lr, gen)
        self.out_net.eval()
        return self

    def score(self, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
        """Anomaly score per row (higher = more unusual). Same units as the output autoencoder's RMSE."""
        X = np.asarray(X, dtype=np.float64)
        out = []
        for i in range(0, len(X), chunk):                  # chunked so a whole capture fits in memory
            R = self._group_rmse(self._scale(X[i:i + chunk])) / self.r_hi
            out.append(_rmse(self.out_net, R).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)
