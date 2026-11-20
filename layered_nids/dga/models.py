"""DGA models behind one interface: fit(names, y, val=None) and predict_proba(names) -> P(generated).

  EntropyRule  - one number (character entropy). The rule-style baseline: no learning.
  LexicalLR    - logistic regression on 15 hand-built lexical features. Linear baseline (Unit 1).
  NgramLR      - character n-gram TF-IDF + logistic regression with L2. Strong classical model.
  LexicalMLP   - small MLP (PyTorch) on the same 15 features, L2 via weight decay (Unit 5).
  CharCNN      - character CNN (PyTorch): learns its own n-gram detectors from raw characters.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layered_nids.dga.features import entropy, lexical_matrix


class DGAModel:
    name = "base"
    calibrated = True          # False when scores are not probabilities (rules)

    def fit(self, names, y, val=None):
        raise NotImplementedError

    def predict_proba(self, names) -> np.ndarray:
        raise NotImplementedError


class EntropyRule(DGAModel):
    name = "entropy_rule"
    calibrated = False

    def fit(self, names, y, val=None):
        return self

    def predict_proba(self, names):
        return np.array([entropy(n) for n in names], dtype=np.float64) / 5.0   # ~[0, 1]; ranking is all that is used


class LexicalLR(DGAModel):
    name = "lexical_lr"

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=500, random_state=seed)

    def fit(self, names, y, val=None):
        self.clf.fit(self.scaler.fit_transform(lexical_matrix(names)), y)
        return self

    def predict_proba(self, names):
        return self.clf.predict_proba(self.scaler.transform(lexical_matrix(names)))[:, 1]


class NgramLR(DGAModel):
    name = "ngram_lr"

    def __init__(self, ngram_range=(2, 4), C: float = 4.0, max_features: int = 300_000, seed: int = 0):
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, sublinear_tf=True, min_df=3,
                                   max_features=max_features, dtype=np.float32, lowercase=True)
        self.clf = LogisticRegression(C=C, max_iter=300, solver="lbfgs", random_state=seed)

    def fit(self, names, y, val=None):
        self.clf.fit(self.vec.fit_transform(["^" + n + "$" for n in names]), y)
        return self

    def predict_proba(self, names):
        return self.clf.predict_proba(self.vec.transform(["^" + n + "$" for n in names]))[:, 1]


# ---------------------------------------------------------------------------------------------- PyTorch models
def _torch():
    import torch
    return torch


class _TorchBinary(DGAModel):
    """Shared training loop: Adam, BCE, best epoch by validation loss when `val` is given."""
    epochs = 8
    batch = 512
    lr = 2e-3
    weight_decay = 1e-4

    def __init__(self, seed: int = 0, epochs: int | None = None, threads: int | None = None):
        self.seed = seed
        if epochs is not None:
            self.epochs = epochs
        self.threads = threads
        self.net = None
        self.history: list[dict] = []

    # subclasses implement
    def _encode(self, names, fit: bool = False):
        raise NotImplementedError

    def _build(self, x):
        raise NotImplementedError

    def fit(self, names, y, val=None):
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x = self._encode(names, fit=True)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
        self.net = self._build(x)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        xv = yv = None
        if val is not None:
            xv, yv = self._encode(val[0]), torch.as_tensor(np.asarray(val[1], dtype=np.float32))
        best, best_state = np.inf, None
        n = len(yt)
        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, self.batch):
                idx = torch.as_tensor(order[i:i + self.batch])
                opt.zero_grad()
                loss = loss_fn(self.net(x[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            rec = {"epoch": ep + 1, "train_loss": tot / n}
            if xv is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(self.net(xv).squeeze(-1), yv))
                rec["val_loss"] = vl
                if vl < best:
                    best = vl
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            self.history.append(rec)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, names):
        torch = _torch()
        x = self._encode(names)
        out = []
        self.net.eval()
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                out.append(torch.sigmoid(self.net(x[i:i + 4096]).squeeze(-1)).numpy())
        return np.concatenate(out).astype(np.float64)


class LexicalMLP(_TorchBinary):
    name = "lexical_mlp"
    weight_decay = 1e-4

    def __init__(self, hidden=(64, 32), **kw):
        super().__init__(**kw)
        self.hidden = hidden
        self.scaler = StandardScaler()

    def _encode(self, names, fit=False):
        torch = _torch()
        m = lexical_matrix(names)
        m = self.scaler.fit_transform(m) if fit else self.scaler.transform(m)
        return torch.as_tensor(m.astype(np.float32))

    def _build(self, x):
        torch = _torch()
        layers, d = [], x.shape[1]
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        return torch.nn.Sequential(*layers)


VOCAB = {c: i + 2 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789-._")}   # 0 = pad, 1 = unknown
MAXLEN = 64


class CharCNN(_TorchBinary):
    name = "char_cnn"
    weight_decay = 1e-5

    def __init__(self, emb: int = 32, filters: int = 64, kernels=(3, 4, 5), dropout: float = 0.3, **kw):
        super().__init__(**kw)
        self.emb, self.filters, self.kernels, self.dropout = emb, filters, kernels, dropout

    def _encode(self, names, fit=False):
        torch = _torch()
        arr = np.zeros((len(names), MAXLEN), dtype=np.int64)
        for i, n in enumerate(names):
            ids = [VOCAB.get(c, 1) for c in n[:MAXLEN]]
            arr[i, :len(ids)] = ids
        return torch.as_tensor(arr)

    def _build(self, x):
        torch = _torch()
        nn = torch.nn
        emb, filters, kernels, dropout = self.emb, self.filters, self.kernels, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(len(VOCAB) + 2, emb, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels])
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(filters * len(kernels), 1)

            def forward(self, ids):
                h = self.emb(ids).transpose(1, 2)                       # (batch, emb, length)
                pooled = [torch.relu(c(h)).amax(dim=2) for c in self.convs]   # strongest match of each detector
                return self.out(self.drop(torch.cat(pooled, dim=1)))

        return Net()


REGISTRY = {c.name: c for c in (EntropyRule, LexicalLR, NgramLR, LexicalMLP, CharCNN)}
