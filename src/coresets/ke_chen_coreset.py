import math
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans


class KeChenCoreset:
    def __init__(self, m, k=8, seed=42):
        self.m = int(m)
        self.k = int(k)
        self.seed = seed
        self.indices = None
        self.weights = None


    # Phase 1: fast bicriteria clustering (sklearn optimized)
    def _bicriteria(self, X):
        kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            random_state=self.seed,
            batch_size=4096,
            n_init=10,
            max_iter=50
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_


    # Phase 2: fast assignment (C-optimized sklearn routine)
    def _assign(self, X, centers):
        return pairwise_distances_argmin(X, centers)

    # Main coreset construction
    def generate(self, data):

        X = np.asarray(data, dtype=np.float32)
        n = X.shape[0]

        rng = np.random.default_rng(self.seed)

        # 1. Bicriteria centers
        A = self._bicriteria(X)

        labels = self._assign(X, A)

        diffs = X - A[labels]
        sq_dists = np.sum(diffs * diffs, axis=1)

        # k-means objective uses squared distances, keep squared
        dists = sq_dists

        nu_A = float(np.sum(dists))

        # 2. Ring partition
        R = max(nu_A / max(n, 1), 1e-12)
        phi = max(1, math.ceil(math.log2(max(n, 2))))

        ring_sets = {}

        for i in range(n):

            dist = dists[i]

            if dist <= R:
                j = 0
            else:
                # use sqrt only for indexing stability
                j = min(phi, max(1, math.ceil(math.log2((dist + 1e-12) / (R + 1e-12)))))

            key = (int(labels[i]), j)

            if key not in ring_sets:
                ring_sets[key] = []

            ring_sets[key].append(i)

        
        # 3. Adaptive sampling
        items = list(ring_sets.items())
        ring_sizes = np.array([len(v) for _, v in items], dtype=np.float64)

        total = ring_sizes.sum()

        ring_budgets = np.maximum(
            1,
            np.floor(self.m * ring_sizes / total).astype(int)
        )

        while ring_budgets.sum() > self.m:
            ring_budgets[np.argmax(ring_budgets)] -= 1

        while ring_budgets.sum() < self.m:
            ring_budgets[np.argmax(ring_sizes)] += 1

        # 4. Sampling
        S_list = []
        W_list = []
        I_list = []

        for budget, (_, idx_list) in zip(ring_budgets, items):

            idx = np.asarray(idx_list)

            size = len(idx)

            if size == 0:
                continue

            b = min(size, budget)

            chosen = rng.choice(size, size=b, replace=False)

            sel = idx[chosen]

            S_list.append(X[sel])

            # weight = ring scaling
            W_list.append(np.full(b, size / b, dtype=np.float32))

            I_list.append(sel)

        S = np.vstack(S_list)
        W = np.concatenate(W_list)
        I = np.concatenate(I_list)

        self.indices = I.astype(np.int32)
        self.weights = W.astype(np.float32)

        return S, W


def final_kmeans(X, k, w):
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(X, sample_weight=w)
    return km.cluster_centers_


from sklearn.cluster import MiniBatchKMeans

def fit_final_kmeans(S, W, k, seed):
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
        max_iter=100,
        batch_size=2048
    )

    model.fit(S, sample_weight=W)

    return model.cluster_centers_