import numpy as np

def _d2(X, C):
    return (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]

class KMeansNP:
    """Plain numpy K-Means (k-means++ init, Lloyd iterations). sklearn's version is blocked by Windows policy."""
    def __init__(self, k, iters=50, seed=0, tol=1e-4):
        self.k, self.iters, self.seed, self.tol = k, iters, seed, tol
    def fit(self, X):
        rng = np.random.RandomState(self.seed); n = len(X)
        C = [X[rng.randint(n)]]
        for _ in range(1, self.k):
            d = np.maximum(_d2(X, np.array(C)).min(1), 0); C.append(X[rng.choice(n, p=d / d.sum())])
        C = np.array(C); prev = None
        for _ in range(self.iters):
            d = _d2(X, C); lab = d.argmin(1); inertia = d[np.arange(n), lab].sum()
            for j in range(self.k):
                m = lab == j
                if m.any(): C[j] = X[m].mean(0)
            if prev is not None and abs(prev - inertia) <= self.tol * prev: break
            prev = inertia
        self.centers_, self.inertia_ = C, inertia; return self
    def distance(self, X):
        return np.sqrt(np.maximum(_d2(X, self.centers_).min(1), 0))
    def predict(self, X):
        return _d2(X, self.centers_).argmin(1)

def silhouette(X, lab, n=4000, seed=0):
    rng = np.random.RandomState(seed); i = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs, ls = X[i], lab[i]; D = np.sqrt(np.maximum(_d2(Xs, Xs), 0)); s = []
    ks = np.unique(ls)
    if len(ks) < 2: return float('nan')
    for a in range(len(Xs)):
        same = ls == ls[a]; same[a] = False
        if not same.any(): continue
        A = D[a, same].mean(); B = min(D[a, ls == k].mean() for k in ks if k != ls[a])
        s.append((B - A) / max(A, B))
    return float(np.mean(s))
