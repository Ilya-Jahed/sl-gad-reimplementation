# SL-GAD Sensitivity Analysis (`plot_sensitivity.py`)

Reproduces the standard AUC-vs-hyperparameter figure for SL-GAD:

| Panel | Hyperparameter | Default sweep values |
|---|---|---|
| (a) | Evaluation rounds (`auc_test_rounds`) | 1, 5, 10, 20, 40, 80, 160, 320 |
| (b) | Subgraph size (`subgraph_size`) | 2, 4, 6, 8, 10, 12, 14 |
| (c) | Hidden dimension (`embedding_dim`) | 2, 4, 8, 16, 32, 64, 128, 256 |
| (d) | Negative ratio (`negsamp_ratio`) | 1, 2, 4, 8, 16, 32, 64, 128 |

For every dataset you pass in, the script trains and evaluates the model
across each range above, then plots all datasets together in a 2x2 figure
matching the usual paper-style layout (one colored line per dataset).

This file does **not** modify `main.py`, `src/trainer.py`, `src/model.py`,
or `src/evaluator.py` — it only imports and reuses them. Drop
`plot_sensitivity.py` in the repo root, next to `main.py`.

---

## 1. Setup

### Requirements
Same as the rest of the repo, plus `matplotlib`:

```bash
pip install torch dgl scikit-learn networkx scipy matplotlib
```

### Data
Datasets must exist at `./data/<dataset>.mat`, exactly like `main.py` expects
(e.g. `./data/cora.mat`, `./data/citeseer.mat`, `./data/BlogCatalog.mat`).

---

## 2. How the two "modes" work

- **Panel (a) — Evaluation rounds**: cheap. The model is trained **once**,
  then `Evaluator.evaluate()` is re-run with different `auc_test_rounds`
  values on the same trained model — no retraining.
- **Panels (b), (c), (d)**: expensive. Each value changes the model's
  architecture or input shape, so a **full training run** happens for every
  point on every dataset (e.g. 8 values × 2 datasets = 16 full trainings for
  panel (c) alone).

Total full trainings for all 4 panels, per dataset:
`1 (a) + 7 (b) + 8 (c) + 8 (d) = 24`

---

## 3. Quick local test (CPU, sanity check only)

Use `--quick` to cap epochs/patience so you can confirm the pipeline runs
end-to-end and produces a sane-looking plot, without waiting for a real
result:

```bash
python plot_sensitivity.py \
  --datasets cora \
  --panels eval_rounds \
  --device cpu \
  --quick
```

If that works, try adding one expensive panel to confirm retraining also
works:

```bash
python plot_sensitivity.py \
  --datasets cora \
  --panels eval_rounds subgraph_size \
  --device cpu \
  --quick
```

`--quick` results are **not meaningful** (too few epochs to converge) — only
use them to check nothing crashes and the figure/cache files are created
correctly.

---

## 4. Full run (GPU — e.g. Colab / university A100)

Drop `--quick`, pick real datasets, and point `--device` at your GPU:

```bash
python plot_sensitivity.py \
  --datasets cora citeseer BlogCatalog \
  --panels eval_rounds subgraph_size hidden_dim negative_ratio \
  --device cuda:0
```

### Running in Jupyter (won't die if the kernel disconnects)
```python
import subprocess
log = open('sensitivity.log', 'w')
proc = subprocess.Popen(
    ['python', '-u', 'plot_sensitivity.py',
     '--datasets', 'cora', 'citeseer',
     '--panels', 'eval_rounds', 'subgraph_size', 'hidden_dim', 'negative_ratio',
     '--device', 'cuda:0'],
    stdout=log, stderr=subprocess.STDOUT
)
print('PID:', proc.pid)
```
Check progress anytime with:
```python
!tail -n 30 sensitivity.log
```

---

## 5. CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `--datasets` | *(required)* | One or more dataset names, matching `data/<name>.mat` |
| `--panels` | all four | Which panels to compute: `eval_rounds subgraph_size hidden_dim negative_ratio` |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |
| `--seed` | `1` | Random seed |
| `--quick` | off | Caps epochs/patience to ~15 for a fast smoke test |
| `--cache` | `results/sensitivity_cache.json` | Where computed AUCs are saved |
| `--out` | `results/sensitivity_plot.png` | Output figure path |
| `--plot_only` | off | Skip training entirely, just re-plot from `--cache` |

---

## 6. Outputs

- `results/sensitivity_cache.json` — raw AUC numbers per panel/dataset/value.
  Keep this file; you can re-plot from it anytime without retraining:
  ```bash
  python plot_sensitivity.py --datasets cora citeseer --plot_only \
    --cache results/sensitivity_cache.json --out results/replot.png
  ```
- `results/sensitivity_plot.png` — the 2x2 figure.
- `checkpoints/exp_sens_<dataset>_<tag>.pkl` — one checkpoint per training
  run (same mechanism `main.py` uses). These can be deleted after the run if
  disk space matters.

---

## 7. Judging whether the result looks "good"

Once you have a plot, sanity checks worth doing before trusting it:

- **Panel (a)** should flatten out (AUC stabilizing) as rounds increase —
  if it's still trending up/down sharply at 320 rounds, variance is high.
- **Panels (b)/(c)/(d)** should be roughly smooth curves, not jagged —
  jaggedness usually means too few training epochs (check you didn't leave
  `--quick` on) or too much run-to-run variance from a single seed.
- Compare the AUC *level* (not just the shape) to published SL-GAD/CoLA
  numbers for the same dataset — if it's far below typical reported
  values (e.g. Cora usually ~0.85+, Pubmed ~0.95+), something in
  training config (epochs, lr, patience) likely needs adjustting before
  the sweep is trustworthy.

---

## 8. Known cost caveat

The bottleneck is usually **not** the GPU — it's `generate_rwr_subgraph()`
in `src/dataset.py`, which runs on CPU once per epoch (twice, for the two
views) over the whole graph. A stronger GPU won't help much unless the CPU
behind it is also fast. Larger datasets (Pubmed, BlogCatalog) and more
epochs multiply this cost directly.