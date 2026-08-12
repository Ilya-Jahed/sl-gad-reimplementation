# SL-GAD: Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection

> A clean, modular reimplementation of the **SL-GAD** framework for unsupervised graph anomaly detection on attributed graphs.  
> Based on: *IEEE Transactions on Knowledge and Data Engineering, 2021*

---

## 📌 Project Overview

SL-GAD detects anomalies in attributed graphs by leveraging two parallel self-supervised views:

| Module | Role |
|---|---|
| **Generative** | Reconstructs target-node attributes from its contextual subgraph neighbours |
| **Contrastive** | Contrasts a target node's embedding against its subgraph context to detect structural anomalies |

The two anomaly scores are fused at inference time, making SL-GAD robust to both **attribute** and **structural** anomalies.

---

## 📂 Project Structure

```
sl-gad-reimplementation/
├── data/                        # .mat datasets (Cora, CiteSeer, BlogCatalog, …)
├── checkpoints/                 # Saved model checkpoints (auto-created at training)
├── docs/
│   ├── dataset_pipeline.md      # src/dataset.py — data loading & RWR sampling
│   ├── model_architecture.md    # src/model.py — GCN, Discriminator, readout
│   └── training_pipeline.md     # src/trainer.py, evaluator.py, main.py
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Data loading, normalization & RWR subgraph sampling
│   ├── model.py                 # GCN encoder/decoder, Discriminator, SL-GAD Model
│   ├── trainer.py               # Batch construction, joint loss, early stopping
│   └── evaluator.py             # Multi-round anomaly scoring & AUC evaluation
├── main.py                      # CLI entry point — runs the full pipeline
├── test_dataset.py              # Smoke-test for src/dataset.py
├── test_model.py                # Smoke-test for src/model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup (Conda)

### Step 1 — Create and activate the environment

```bash
conda create -n slgad python=3.7 -y
conda activate slgad
```

> **Why Python 3.7?** The source targets `torch==1.8.1` and `dgl==0.4.1`, both built against Python 3.7.

---

### Step 2 — Install PyTorch 1.8.1

> ⚠️ **`torch==1.8.1` is NOT on PyPI.** You must use the `-f` flag below.  
> ⚠️ **Do NOT include `torchaudio`** — it conflicts with `torch==1.8.1`.

**CPU only (recommended for getting started)**
```bash
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

**CUDA 11.1**
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

---

### Step 3 — Install DGL 0.4.1

> ⚠️ **Must be exactly `0.4.1`**. The RWR sampling in `dataset.py` uses `dgl.contrib.sampling.random_walk_with_restart`, which was removed in later DGL versions.

**CPU only**
```bash
pip install dgl==0.4.1
```

**CUDA 11.1**
```bash
pip install dgl-cu111==0.4.1
```

---

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Verify

```bash
python -c "import torch, dgl; print('torch:', torch.__version__); print('dgl:', dgl.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 🧪 Smoke Tests

| Test | What it validates |
|---|---|
| `python test_dataset.py` | All functions in `src/dataset.py` using a real `.mat` file |
| `python test_model.py` | All model components using synthetic random tensors |

```bash
# Dataset test (default: cora.mat)
python test_dataset.py

# Model test
python test_model.py
```

---

## 🚀 Usage

### Minimal run

```bash
python main.py --expid 1 --dataset cora
```

### Common examples

```bash
# BlogCatalog on GPU
python main.py --expid 2 --dataset BlogCatalog --device cuda:0

# 3 independent runs for statistical reliability
python main.py --expid 3 --dataset ACM --runs 3

# Contrastive score only (disable generative module at eval)
python main.py --expid 4 --dataset cora --alpha 1.0 --beta 0.0

# Faster eval for debugging (fewer inference rounds)
python main.py --expid 5 --dataset cora --auc_test_rounds 10
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--expid` | *(required)* | Experiment ID for checkpoint naming |
| `--dataset` | `BlogCatalog` | Dataset name (must match `data/<name>.mat`) |
| `--device` | `cuda:0` | Compute device; auto-falls back to CPU |
| `--runs` | `1` | Number of independent training runs |
| `--embedding_dim` | `64` | Hidden embedding dimension |
| `--readout` | `avg` | Pooling: `avg`, `max`, `min`, `weighted_sum` |
| `--alpha` | `1.0` | Contrastive loss/score weight |
| `--beta` | `0.6` | Generative loss/score weight |
| `--auc_test_rounds` | `256` | Inference rounds for stable AUC |

---

## 📊 Supported Datasets

| Dataset | Nodes | Edges | Features | Anomaly Rate |
|---|---|---|---|---|
| Cora | 2,708 | 5,429 | 1,433 | ~5.5% |
| CiteSeer | 3,327 | 4,732 | 3,703 | — |
| Pubmed | 19,717 | 44,338 | 500 | — |
| BlogCatalog | 5,196 | 171,743 | 8,189 | ~5% |
| Flickr | 7,575 | 239,738 | 12,047 | ~5% |
| ACM | 16,484 | 71,980 | 8,337 | ~5% |

Place `.mat` files in the `data/` directory before running.

---

## 📈 Verified Results

| Dataset | This Implementation | Paper (SL-GAD) |
|---|---|---|
| Cora | **0.9167** | ~0.90–0.93 |

---

## 📖 Documentation

| Document | Contents |
|---|---|
| [`docs/dataset_pipeline.md`](docs/dataset_pipeline.md) | `src/dataset.py` — data loading, adjacency normalization, RWR subgraph sampling |
| [`docs/model_architecture.md`](docs/model_architecture.md) | `src/model.py` — GCN encoder/decoder, Discriminator with negative sampling, readout modes, subgraph expansion trick |
| [`docs/training_pipeline.md`](docs/training_pipeline.md) | `src/trainer.py`, `src/evaluator.py`, `main.py` — batch construction, dual loss, multi-round AUC evaluation |

---

## ✅ Implementation Status

- [x] `src/dataset.py` — data loading, normalization, RWR subgraph sampling
- [x] `src/model.py` — GCN encoder/decoder, 4 readout modes, bilinear discriminator with cyclic negative sampling
- [x] `src/trainer.py` — $K \to K+1$ subgraph expansion, joint BCE + MSE loss, early stopping
- [x] `src/evaluator.py` — multi-round stochastic inference, MinMax score fusion, ROC-AUC
- [x] `main.py` — full CLI pipeline
- [x] `test_dataset.py` — dataset smoke test
- [x] `test_model.py` — model architecture smoke test

---

## 📜 Reference

```bibtex
@article{zheng2021generative,
  title     = {Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection},
  author    = {Zheng, Yu and Jin, Ming and Liu, Yixin and Chi, Lianhua and Phan, Khoa T. and Chen, Yi-Ping Phoebe},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  year      = {2021},
  doi       = {10.1109/TKDE.2021.3119326}
}
```

---

## 📜 License

This project is for research and educational purposes. Please cite the original paper if you use this code.