import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class EGBCoreset:
    """Exponential Grid-Based coreset for k-means.

    Binary-searches epsilon so that |Q| ≈ the target size m.
    Tries c in {0.3, 1.0, 3.0} and picks whichever (epsilon, c) gets
    closest to the target.  A per-seed refinement re-tunes epsilon for
    each actual A so the match stays tight across trials.
    """

    INITIAL_K_FACTOR = 3
    _param_cache: dict = {}

    def __init__(self, m, k=None):
        self.m = m
        self.k = k
        self.weights = None
        self.indices = None
        self.sensitivities = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _assign(P, A):
        """Chunked nearest-center assignment to avoid OOM on large data."""
        n = len(P)
        nearest = np.empty(n, dtype=np.int64)
        min_sq = np.empty(n, dtype=np.float64)
        chunk = 50_000
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            d2 = cdist(P[i:j], A, "sqeuclidean")
            nearest[i:j] = np.argmin(d2, axis=1)
            min_sq[i:j] = d2[np.arange(j - i), nearest[i:j]]
        return nearest, min_sq

    @staticmethod
    def _build_inv(P, A, c, nearest, min_sq):
        """Build EGB invariants from pre-computed assignments."""
        n, d = P.shape
        mu = float(np.sum(min_sq))
        R = np.sqrt(mu / (c * n))
        if R == 0:
            return None

        diffs = P - A[nearest]
        Linf = np.max(np.abs(diffs), axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.maximum(1, np.ceil(np.log2(Linf / R)))

        M = int(np.ceil(2.0 * np.log2(c * n)))
        j = np.clip(j, 0, M).astype(np.int64)

        return {
            "n": n, "d": d, "R": R, "c": c,
            "nearest": nearest,
            "j": j,
            "diffs": diffs,
            "pow2j": 2.0 ** j,
        }

    @staticmethod
    def _sigs(inv, epsilon):
        rj = (epsilon * inv["R"] * inv["pow2j"]) / (10.0 * inv["c"] * inv["d"])
        cells = np.floor(inv["diffs"] / rj[:, None]).astype(np.int64)
        return np.hstack([inv["nearest"][:, None], inv["j"][:, None], cells])

    @classmethod
    def _count_fast(cls, inv, epsilon):
        """Fast unique count via row hashing (for binary search)."""
        sigs = cls._sigs(inv, epsilon)
        h = sigs[:, 0].copy()
        for col in range(1, sigs.shape[1]):
            np.multiply(h, 6364136223846793005, out=h)
            np.add(h, sigs[:, col], out=h)
        return len(np.unique(h))

    @classmethod
    def _extract(cls, P, inv, epsilon):
        """Full extraction: unique signatures → representative points."""
        sigs = cls._sigs(inv, epsilon)
        _, idx, cnt = np.unique(sigs, axis=0, return_index=True, return_counts=True)
        return P[idx], idx, cnt.astype(float)

    @classmethod
    def _binary_search(cls, inv, target, lo=1e-4, hi=500.0, iters=20):
        best_eps, best_diff = np.sqrt(lo * hi), float("inf")
        for _ in range(iters):
            mid = np.sqrt(lo * hi)
            size = cls._count_fast(inv, mid)
            diff = abs(size - target)
            if diff < best_diff:
                best_diff = diff
                best_eps = mid
            if size == target:
                break
            elif size > target:
                lo = mid
            else:
                hi = mid
        return best_eps, best_diff

    @classmethod
    def _find_best(cls, P, A, target):
        """Try several c values, binary-search epsilon for each, pick best."""
        nearest, min_sq = cls._assign(P, A)
        best_eps, best_c, best_diff = 1.0, 1.0, float("inf")

        for c in [0.3, 1.0, 3.0]:
            inv = cls._build_inv(P, A, c, nearest, min_sq)
            if inv is None:
                continue

            eps, diff = cls._binary_search(inv, target)

            if diff < best_diff:
                best_diff = diff
                best_eps, best_c = eps, c

            if diff == 0:
                break

        return best_eps, best_c

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(self, data):
        n, d = data.shape
        k = self.k if self.k is not None else max(2, int(np.sqrt(self.m)))
        n_init = min(k * self.INITIAL_K_FACTOR, n - 1)

        fp = hash(data[: min(3, n)].tobytes())
        cache_key = (self.m, k, n, d, fp)

        if cache_key not in self._param_cache:
            rng = np.random.get_state()
            A_search = KMeans(
                n_clusters=n_init, n_init=5, max_iter=100, random_state=42
            ).fit(data).cluster_centers_
            np.random.set_state(rng)
            eps, c = self._find_best(data, A_search, self.m)
            self._param_cache[cache_key] = (eps, c)

        eps_cached, c = self._param_cache[cache_key]

        rs = np.random.randint(0, 2**31)
        A = KMeans(
            n_clusters=n_init, n_init=5, max_iter=100, random_state=rs
        ).fit(data).cluster_centers_

        nearest, min_sq = self._assign(data, A)
        inv = self._build_inv(data, A, c, nearest, min_sq)

        if inv is None:
            return self._fallback(data, n)

        # Refine epsilon for this specific A (15 quick iterations)
        lo = max(eps_cached * 0.01, 1e-4)
        hi = min(eps_cached * 100.0, 500.0)
        epsilon, _ = self._binary_search(inv, self.m, lo=lo, hi=hi, iters=15)

        Q, indices, weights = self._extract(data, inv, epsilon)

        if len(Q) <= k:
            return self._fallback(data, n)

        self.indices = indices
        self.weights = weights
        self.sensitivities = None
        return Q

    def _fallback(self, data, n):
        m = min(self.m, n)
        self.indices = np.random.choice(n, size=m, replace=False)
        self.weights = np.full(m, n / m)
        self.sensitivities = None
        return data[self.indices]
