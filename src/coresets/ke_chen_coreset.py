import math
import numpy as np


def _kmeans_plus_plus_init(X, k, rng):
    n, d = X.shape
    centers = np.empty((k, d))
    centers[0] = X[rng.integers(n)]
    for i in range(1, k):
        diffs = X[:, np.newaxis, :] - centers[:i][np.newaxis, :, :]
        min_sq = np.sum(diffs ** 2, axis=2).min(axis=1)
        probs = min_sq / min_sq.sum()
        centers[i] = X[rng.choice(n, p=probs)]
    return centers


def _assign_clusters(X, centers):
    diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diffs ** 2, axis=2).argmin(axis=1)


def _kmeans_plus_plus(X, k, weights=None, max_iter=50, seed=None):
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = np.ones(len(X))
    centers = _kmeans_plus_plus_init(X, k, rng)
    prev_cost = np.inf
    for _ in range(max_iter):
        labels = _assign_clusters(X, centers)
        new_centers = np.empty_like(centers)
        for c in range(k):
            mask = labels == c
            new_centers[c] = (np.average(X[mask], axis=0, weights=weights[mask])
                              if mask.any() else X[rng.integers(len(X))])
        cost = float(np.dot(weights,
                            np.sum((X - new_centers[_assign_clusters(X, new_centers)]) ** 2,
                                   axis=1)))
        centers = new_centers
        if prev_cost > 0 and abs(prev_cost - cost) / prev_cost < 1e-6:
            break
        prev_cost = cost
    return centers


def _build_ring_partition(X, A, nu_A, beta=1.0):
    n = len(X)
    R = max(nu_A / (beta * n), 1e-10) if n > 0 else 1e-10
    phi = math.ceil(math.log2(beta * n)) if n > 1 else 1
    labels = _assign_clusters(X, A)
    ring_sets = {}
    for idx in range(n):
        i = int(labels[idx])
        dist = float(np.linalg.norm(X[idx] - A[i]))
        j = 0 if dist <= R else min(phi, max(1, math.ceil(math.log2(dist / R))))
        ring_sets.setdefault((i, j), []).append(idx)
    return ring_sets, R


def _compute_sample_size(k, epsilon, n, lam=0.1, beta=1.0, c=0.1):
    if n <= 1:
        return 1
    s = c * (beta ** 2) / (epsilon ** 2) * (k * math.log(n) + math.log(1.0 / lam))
    return max(1, math.ceil(s))


def _epsilon_for_target_size(X, k, target_m, seed=0, tol=0.05, max_iter=30):
    lo, hi = 0.01, 10.0
    best_eps = 0.3
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        s = _compute_sample_size(k, mid, len(X))
        # Rough size estimate: n_rings * s  (rings ≈ k * log2(n))
        n_rings = k * max(1, math.ceil(math.log2(max(len(X), 2))))
        est_size = min(len(X), n_rings * s)
        if est_size > target_m:
            lo = mid
        else:
            hi = mid
        best_eps = mid
        if abs(est_size - target_m) / max(target_m, 1) < tol:
            break
    return best_eps


class KeChenCoreset:
    def __init__(self, m, k=8, seed=42):
        self.m    = m
        self.k    = k
        self.seed = seed

        self.indices = None
        self.weights = None

    def generate(self, data):
        X = np.asarray(data, dtype=float)
        n, d = X.shape
        k = self.k
        rng = np.random.default_rng(self.seed)

        # Phase 1: approximate centers (O(nk), same asymptotic as Lloyd)
        A = _kmeans_plus_plus(X, k, max_iter=50, seed=self.seed)

        labels = _assign_clusters(X, A)
        dists  = np.linalg.norm(X - A[labels], axis=1)
        nu_A   = float(dists.sum())

        # Phase 2: ring partition
        ring_sets, R = _build_ring_partition(X, A, nu_A)

        # Phase 3: sample s points from each ring, weighted by ring_size / s
        # Tune epsilon so total coreset size ≈ self.m
        eps = _epsilon_for_target_size(X, k, self.m, seed=self.seed)
        s   = _compute_sample_size(k, eps, n)

        coreset_pts, coreset_wts, coreset_idx = [], [], []

        for (i, j), idx_list in ring_sets.items():
            size = len(idx_list)
            pts  = X[idx_list]
            if size <= s:
                coreset_pts.append(pts)
                coreset_wts.append(np.ones(size))
                coreset_idx.extend(idx_list)
            else:
                chosen = rng.choice(size, size=s, replace=True)
                coreset_pts.append(pts[chosen])
                coreset_wts.append(np.full(s, size / s))
                coreset_idx.extend([idx_list[c] for c in chosen])

        points  = np.vstack(coreset_pts)
        weights = np.concatenate(coreset_wts)

        # indices: best-effort (may have duplicates from sampling with replacement)
        self.indices = np.array(coreset_idx, dtype=int)
        self.weights = weights

        return points, weights