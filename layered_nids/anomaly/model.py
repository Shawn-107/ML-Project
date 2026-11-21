"""NIDS anomaly model: an ensemble of small autoencoders trained on normal traffic only.

Idea (from the Kitsune paper, own implementation and own choices):
  1. Features that move together share a small autoencoder (correlation clustering, at most `max_group` features each).
     Small networks are cheap and one odd corner of the feature space is not averaged away by the other features.
  2. Each autoencoder learns to rebuild its group on NORMAL data; its reconstruction error (RMSE) says how unusual the group is.
  3. The group errors are combined into one anomaly score. Two ways, chosen by validation, not by taste:
       "output_ae"  a final autoencoder over the vector of group errors (what the paper does); score = its RMSE
       "tail_sum"   NEW: for every group, how rare is this error among normal calibration errors (empirical tail);
                    score = sum of -log(tail probability) over the `topk` rarest groups. No extra network, explainable
                    ("these 3 feature groups are each rarer than 1 in 1000"), and the groups responsible are known.

Pre-processing (own choice): non-negative features get log1p (sizes, counts and rates are heavy-tailed), then every feature
is scaled so its 0.5% and 99.5% training quantiles map to 0 and 1. Values outside that range are NOT clipped to it (only
capped at +-30 to stay finite), so a feature far outside normal makes a large error.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from layered_nids.packets.ensemble import correlation_groups

CAP = 30.0


class Prep:
    def fit(self, X: np.ndarray) -> "Prep":
        X = np.asarray(X, dtype=np.float64)
        self.log = X.min(0) >= 0.0
        Z = self._log(X)
        self.lo = np.quantile(Z, 0.005, axis=0)
        hi = np.quantile(Z, 0.995, axis=0)
        self.span = np.where(hi - self.lo > 1e-9, hi - self.lo, 1.0)
        return self

    def _log(self, X: np.ndarray) -> np.ndarray:
        Z = np.array(X, dtype=np.float64, copy=True)
        Z[:, self.log] = np.log1p(np.maximum(Z[:, self.log], 0.0))
        return Z

    def transform(self, X: np.ndarray) -> torch.Tensor:
        Z = (self._log(np.asarray(X, dtype=np.float64)) - self.lo) / self.span
        return torch.as_tensor(np.clip(Z, -CAP, CAP).astype(np.float32))

    def state(self) -> dict:
        return {"log": self.log.tolist(), "lo": self.lo.tolist(), "span": self.span.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "Prep":
        p = cls()
        p.log, p.lo, p.span = np.asarray(s["log"], bool), np.asarray(s["lo"]), np.asarray(s["span"])
        return p


class AE(nn.Module):
    def __init__(self, n_in: int, ratio: float):
        super().__init__()
        h = max(1, math.ceil(n_in * ratio))
        self.enc = nn.Linear(n_in, h)
        self.dec = nn.Linear(h, n_in)

    def forward(self, x):
        return torch.sigmoid(self.dec(torch.sigmoid(self.enc(x))))


def rmse(model: AE, x: torch.Tensor, batch: int = 16384) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]
            out.append(torch.sqrt(((model(xb) - xb) ** 2).mean(1)))
    return torch.cat(out) if out else torch.zeros(0)


def train(model: AE, x: torch.Tensor, epochs: int, batch: int, lr: float, gen: torch.Generator, noise: float = 0.0) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            xb = x[order[i:i + batch]]
            inp = xb + noise * torch.randn(xb.shape, generator=gen) if noise > 0 else xb
            opt.zero_grad()
            loss = ((model(inp) - xb) ** 2).mean()
            loss.backward()
            opt.step()


class AnomalyEnsemble:
    def __init__(self, max_group: int = 10, hidden_ratio: float = 0.75, epochs: int = 12, batch: int = 256, lr: float = 5e-3,
                 noise: float = 0.0, score_mode: str = "output_ae", topk: int = 3, seed: int = 0):
        if score_mode not in ("output_ae", "tail_sum"):
            raise ValueError("score_mode must be 'output_ae' or 'tail_sum'")
        self.cfg = dict(max_group=max_group, hidden_ratio=hidden_ratio, epochs=epochs, batch=batch, lr=lr, noise=noise,
                        score_mode=score_mode, topk=topk, seed=seed)

    def _group_rmse(self, Z: torch.Tensor) -> torch.Tensor:
        return torch.stack([rmse(net, Z[:, g]) for net, g in zip(self.nets, self.groups)], dim=1)

    def fit(self, X_train: np.ndarray, X_tail: np.ndarray | None = None) -> "AnomalyEnsemble":
        """Train on normal rows. `X_tail` (more normal rows, NOT used for training) gives the empirical tail of every group's
        error for score_mode 'tail_sum'; if omitted the training rows are used (optimistic)."""
        c = self.cfg
        torch.manual_seed(c["seed"])
        gen = torch.Generator().manual_seed(c["seed"])
        X_train = np.asarray(X_train, dtype=np.float64)
        self.prep = Prep().fit(X_train)
        Z = self.prep.transform(X_train)
        self.groups = correlation_groups(X_train, c["max_group"])
        self.nets = []
        for g in self.groups:
            net = AE(len(g), c["hidden_ratio"])
            train(net, Z[:, g], c["epochs"], c["batch"], c["lr"], gen, c["noise"])
            net.eval()
            self.nets.append(net)
        R = self._group_rmse(Z)
        if c["score_mode"] == "output_ae":
            self.r_hi = R.max(0).values.clamp_min(1e-6)
            self.out_net = AE(R.shape[1], c["hidden_ratio"])
            train(self.out_net, R / self.r_hi, c["epochs"], c["batch"], c["lr"], gen)
            self.out_net.eval()
        else:
            ref = self._group_rmse(self.prep.transform(X_tail)) if X_tail is not None else R
            self.tails = np.sort(ref.numpy().astype(np.float64), axis=0)       # per group, ascending
        return self

    def group_errors(self, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
        out = [self._group_rmse(self.prep.transform(X[i:i + chunk])).numpy() for i in range(0, len(X), chunk)]
        return np.concatenate(out) if out else np.zeros((0, len(self.groups)))

    def _score_from_errors(self, R: np.ndarray) -> np.ndarray:
        if self.cfg["score_mode"] == "output_ae":
            Rt = torch.as_tensor((R.astype(np.float32)) / self.r_hi.numpy())
            return rmse(self.out_net, Rt).numpy().astype(np.float64)
        n = len(self.tails)
        s = np.empty_like(R, dtype=np.float64)
        for j in range(R.shape[1]):
            at_least = n - np.searchsorted(self.tails[:, j], R[:, j], side="left")      # normals with error >= this one
            s[:, j] = -np.log((at_least + 1.0) / (n + 1.0))
        k = min(self.cfg["topk"], s.shape[1])
        return np.sort(s, axis=1)[:, -k:].sum(axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per row, higher = more unusual."""
        return self._score_from_errors(self.group_errors(np.asarray(X, dtype=np.float64)))

    def explain(self, x: np.ndarray, names: list[str], top: int = 3) -> list[str]:
        """Which feature groups are the most unusual for ONE row (tail_sum mode): names of the 3 most-different features per group."""
        R = self.group_errors(np.asarray(x, dtype=np.float64).reshape(1, -1))
        order = np.argsort(R[0])[::-1][:top]
        return [", ".join(names[i] for i in self.groups[j][:3]) for j in order]
