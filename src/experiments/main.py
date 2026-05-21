import os
import pandas as pd

from ..coresets import ALGORITHMS
from ..data import DATASETS, load, ROOT
from .experiment import run_grid, centers_to_dataframe, K_VALUES
from .silhouette import find_best_k
from ..visualization.maps import plot_maps
from ..visualization.images import plot_images
from ..visualization.lines import plot_per_algo, plot_algo_compare
from ..visualization.heatmap import plot_heatmap

OUT_ROOT = os.path.join(ROOT, "output")
RATIOS = ["r1", "r2", "r3"]


def save_results(df, out_dir):
    out_path = os.path.join(out_dir, "results.csv")
    df.to_csv(out_path, index=False)


def save_centers(centers, scaler, feature_names, out_dir):
    centers_df = centers_to_dataframe(centers, scaler, feature_names)
    out_path = os.path.join(out_dir, "centers.csv")
    centers_df.to_csv(out_path, index=False)


def plot_cost_ratio_per_algo(df, name, n, d, out_dir):
    per_algo_dir = os.path.join(out_dir, "cost_ratio", "per_algo")
    for algo in ALGORITHMS:
        plot_per_algo(df, name, algo, n, d, per_algo_dir)


def plot_cost_ratio_compare(df, name, n, d, out_dir, k_values=K_VALUES):
    compare_dir = os.path.join(out_dir, "cost_ratio", "compare")
    for k in k_values:
        for ratio in RATIOS:
            plot_algo_compare(df, name, k, ratio, n, d, compare_dir)


def plot_all_heatmaps(df, name, out_dir):
    heatmaps_dir = os.path.join(out_dir, "heatmaps")
    for algo in ALGORITHMS:
        for ratio in RATIOS:
            plot_heatmap(df, name, algo, ratio, heatmaps_dir)


def plot_dataset_specific(P, P_raw, scaler, info, out_dir):
    if info.get("type") == "image":
        plot_images(P, P_raw, scaler, info, ALGORITHMS, os.path.join(out_dir, "image"))
    elif info.get("map"):
        plot_maps(P, P_raw, scaler, info, ALGORITHMS, os.path.join(out_dir, "maps"))


def run_dataset(name):
    P, P_raw, scaler, info = load(name)
    n, d = P.shape
    print(f"[{name}] n={n:,} d={d}", flush=True)

    k_values = list(K_VALUES)
    sil_rows = []

    if info.get("type") != "image":
        best_k, scores = find_best_k(P)
        print(f"  Silhouette scores:", flush=True)
        for k_cand, s in sorted(scores.items()):
            marker = " <-- best" if k_cand == best_k else ""
            print(f"    k={k_cand:3d}: {s:.4f}{marker}", flush=True)
            sil_rows.append({"dataset": name, "k": k_cand, "score": s, "best": k_cand == best_k})
        if best_k not in k_values:
            k_values.append(best_k)
            k_values.sort()

    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    df, centers = run_grid(P, ALGORITHMS, k_values=k_values)

    save_results(df, out_dir)
    save_centers(centers, scaler, info["features"], out_dir)

    plot_cost_ratio_per_algo(df, name, n, d, out_dir)
    plot_cost_ratio_compare(df, name, n, d, out_dir, k_values=k_values)
    plot_all_heatmaps(df, name, out_dir)
    plot_dataset_specific(P, P_raw, scaler, info, out_dir)

    return sil_rows


def _save_silhouette(all_sil):
    if all_sil:
        os.makedirs(OUT_ROOT, exist_ok=True)
        sil_df = pd.DataFrame(all_sil)
        sil_df.to_csv(os.path.join(OUT_ROOT, "silhouette.csv"), index=False)


def main():
    all_sil = []
    for name in DATASETS:
        all_sil.extend(run_dataset(name))
        _save_silhouette(all_sil)


if __name__ == "__main__":
    main()
