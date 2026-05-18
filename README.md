# Coreset Comparison

Compares coreset algorithms (currently lightweight and random) for k-means clustering, across several datasets (tabular and images), several `k` values, and several coreset sizes `|Q|`.

Run it:

```bash
python -m src.experiments.main
```

Outputs land in `output/<dataset>/`.

## The three cost ratios

For each trial we compute three numbers (`C` = centers from full data, `C'` = centers from the coreset):

- **r1 = Cost(Q, C) / Cost(P, C)** — does the coreset's cost match the full cost?
- **r2 = Cost(Q, C') / Cost(P, C')** — same, using coreset-trained centers.
- **r3 = Cost(P, C') / Cost(P, C)** — how good are the coreset's centers on the full data?

Closer to 1.0 is better.

## Files

```
src/
├── coresets/
│   ├── lightweight_coreset.py   importance-sampled coreset
│   ├── random_sampling.py       uniform-random baseline
│   └── __init__.py              ALGORITHMS registry
├── data/
│   └── datasets.py              loads CSVs and images, z-score normalizes
├── experiments/
│   ├── kmeans_cost.py           the k-means objective
│   ├── experiment.py            run_grid + center bookkeeping
│   └── main.py                  entry point: loops datasets, saves CSVs, calls plots
└── visualization/
    ├── lines.py                 cost-ratio line plots
    ├── heatmap.py               k × |Q| heatmaps
    ├── maps.py                  geographic scatter plots
    └── images.py                image-quantization plots
```

## Output

```
output/<dataset>/
├── results.csv                  one row per trial
├── centers.csv                  every k-means center, original units
├── cost_ratio/
│   ├── per_algo/<algo>.png      fix algorithm, vary k & |Q|
│   └── compare/<ratio>_k=<k>.png fix parameters, compare algorithms
├── heatmaps/<algo>_<ratio>.png  k × |Q| grid
└── maps/ or image/              dataset-specific plots
```

## Adding a new algorithm

1. Create `src/coresets/my_algo.py` with a class shaped like `RandomSampling`:
   - `__init__(self, m)` stores target size.
   - `generate(self, data)` returns sampled points, sets `self.indices` and `self.weights`.
2. Add it to `ALGORITHMS` in `src/coresets/__init__.py`.
3. Re-run `python -m src.experiments.main`. It shows up everywhere automatically.
