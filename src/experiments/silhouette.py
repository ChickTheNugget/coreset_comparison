import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CANDIDATE_KS = [2, 3, 4, 6, 8, 10, 12, 16, 24, 32]
SAMPLE_SIZE = 10_000


def find_best_k(P):
    """Sweep candidate k values and return (best_k, {k: score})."""
    n = len(P)
    scores = {}
    for k in CANDIDATE_KS:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=42)
        labels = km.fit_predict(P)
        sil = silhouette_score(P, labels, sample_size=min(SAMPLE_SIZE, n), random_state=42)
        scores[k] = round(float(sil), 4)
    best_k = max(scores, key=scores.get)
    return best_k, scores
