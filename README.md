# Coreset Comparison

Compares coreset algorithms (lightweight, random, EGB, biased, and Ke-Chen) for k-means clustering, across several datasets (tabular and images), several `k` values, and several coreset sizes `|Q|`.

Run it:

```bash
python -m src.experiments.main
```

Outputs land in `output/<dataset>/`.

## Datasets

| Name | Type | Dimensions | Features | n |
|---|---|---|---|---|
| airbnb_sf | tabular (map) | 3D | latitude, longitude, price | 8,041 |
| boris_citibike | tabular (map) | 2D | longitude, latitude | 999,316 |
| martin_spotify | tabular | 4D | energy, valence, danceability, acousticness | 83,371 |
| spotify_6d | tabular | 6D | energy, valence, danceability, acousticness, speechiness, liveness | 114,000 |
| spotify_9d | tabular | 9D | above + instrumentalness, loudness, tempo | 114,000 |
| uber | tabular (map) | 2D | Lon, Lat | 742,264 |
| wholesale | tabular | 6D | Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen | 440 |
| donuts | tabular (map) | 2D | x, y | 1,000,000 |
| birb | image | 3D (RGB) | R, G, B | 1,048,576 |
| arhan_sunset | image | 3D (RGB) | R, G, B | 207,504 |
| image | image | 3D (RGB) | R, G, B | 611,800 |

## The three cost ratios

For each trial we compute three numbers (`C` = centers from full data, `C'` = centers from the coreset):

- **r1 = Cost(Q, C) / Cost(P, C)** — does the coreset's cost match the full cost?
- **r2 = Cost(Q, C') / Cost(P, C')** — same, using coreset-trained centers.
- **r3 = Cost(P, C') / Cost(P, C)** — how good are the coreset's centers on the full data?

Closer to 1.0 is better.

## Experiment grid

- **k values**: `[2, 4, 8, 16, 32, 64]` (plus silhouette-selected best k for tabular datasets)
- **|Q| fractions**: `[0.005%, 0.01%, 0.05%, 0.1%, 0.25%, 0.5%, 1%, 5%, 10%, 25%, 50%]` of n
- **Trials**: 5 seeds per (algorithm, k, |Q|) combination

## Files

```
src/
├── coresets/
│   ├── lightweight_coreset.py   importance-sampled coreset
│   ├── random_sampling.py       uniform-random baseline
│   ├── egb_coreset.py           exponential grid-based coreset
│   ├── biased_coreset.py        biased sampling coreset
│   ├── ke_chen_coreset.py       Ke-Chen ring-partition coreset
│   └── __init__.py              ALGORITHMS registry
├── data/
│   └── datasets.py              loads CSVs and images, z-score normalizes
├── experiments/
│   ├── kmeans_cost.py           the k-means objective
│   ├── experiment.py            run_grid + center bookkeeping
│   ├── silhouette.py            finds best k per dataset via silhouette score
│   └── main.py                  entry point: loops datasets, saves CSVs, calls plots
└── visualization/
    ├── lines.py                 cost-ratio line plots
    ├── heatmap.py               k × |Q| heatmaps
    ├── maps.py                  geographic scatter plots
    └── images.py                image-quantization plots
```

## Output

```
output/
├── silhouette.csv                 silhouette scores for all tabular datasets
└── <dataset>/
    ├── results.csv                one row per trial
    ├── centers.csv                every k-means center, original units
    ├── cost_ratio/
    │   ├── per_algo/<algo>.png    fix algorithm, vary k & |Q|
    │   └── compare/<ratio>_k=<k>.png fix parameters, compare algorithms
    ├── heatmaps/<algo>_<ratio>.png  k × |Q| grid
    └── maps/ or image/            dataset-specific plots
```

## Silhouette-based k selection

For each tabular dataset, a silhouette sweep runs before the main grid to find the best k.
Candidate values: `[2, 3, 4, 6, 8, 10, 12, 16, 24, 32]`. Scores are printed to the terminal
and saved to `output/silhouette.csv`.
If the best k is not already in the default grid (`[2, 4, 8, 16, 32, 64]`), it is added automatically.
Image datasets skip this step.

## Adding a new algorithm

1. Create `src/coresets/my_algo.py` with a class shaped like `RandomSampling`:
   - `__init__(self, m, k=None)` stores target size (and optionally k).
   - `generate(self, data)` returns sampled points, sets `self.indices` and `self.weights`.
2. Add it to `ALGORITHMS` in `src/coresets/__init__.py`.
3. Re-run `python -m src.experiments.main`. It shows up everywhere automatically.

## Adding a new dataset

1. Place the cleaned CSV in `data/real/` (or `data/synthetic/`).
   - Only numeric feature columns + an optional `id` column.
   - No missing values.
2. Add an entry to `DATASETS` in `src/data/datasets.py`:
   - Set `"features"` to the list of column names to use.
   - Set `"map": True` with `"map_dims"` and `"map_labels"` for geographic datasets.
   - Set `"type": "image"` for image datasets.
3. Re-run `python -m src.experiments.main`.
