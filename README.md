# SL-GAD: Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection

> A clean, modular reimplementation of the **SL-GAD** framework for unsupervised graph anomaly detection on attributed graphs.

---

## 📌 Project Overview

SL-GAD detects anomalies in attributed graphs by leveraging two parallel self-supervised views:

| Module | Role |
|---|---|
| **Generative** | Reconstructs target-node attributes from its contextual subgraph |
| **Contrastive** | Contrasts a target node's representation against its subgraph to detect structural anomalies |

The two anomaly scores are fused at inference time, making SL-GAD robust to both **attribute** and **structural** anomalies.

---

## 📂 Project Structure

```
sl-gad-reimplementation/
├── data/                  # .mat datasets (Cora, CiteSeer, BlogCatalog, …)
├── src/
│   ├── __init__.py
│   ├── dataset.py         # Data loading, normalization & RWR subgraph sampling
│   ├── model.py           # Generative + Contrastive model architectures
│   ├── trainer.py         # Training loop & loss computation
│   └── evaluator.py       # AUC-ROC scoring & evaluation utilities
├── main.py                # Entry point – run the full pipeline
├── test_dataset.py        # Smoke-test for src/dataset.py (see Testing section)
├── requirements.txt       # Non-torch dependencies
└── README.md
```

---

## ⚙️ Environment Setup (Conda)

### Step 1 — Create and activate the environment

```bash
conda create -n slgad python=3.7 -y
conda activate slgad
```

> **Why Python 3.7?** The source code targets `torch==1.8.1` and `dgl==0.4.1`, both of which were built against Python 3.7.

---

### Step 2 — Install PyTorch 1.8.1

> ⚠️ **`torch==1.8.1` is NOT on PyPI.** PyTorch hosts its own wheel server, so a plain `pip install torch==1.8.1` will fail with *"No matching distribution found"*. You must use the `-f` flag below.

> ⚠️ **Do NOT include `torchaudio` in this command.** `torchaudio==0.8.0` declares a hard dependency on `torch==1.8.0` (not `1.8.1`), causing an irresolvable conflict. This project does not use torchaudio.

**CPU only (recommended for getting started)**
```bash
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

**CUDA 11.1**
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

**CUDA 10.2**
```bash
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

---

### Step 3 — Install DGL 0.4.1

> ⚠️ **Must match the source code exactly.** The RWR sampling in `dataset.py` uses `dgl.contrib.sampling.random_walk_with_restart`, which only exists in `dgl==0.4.1` and was removed in later versions.

**CPU only**
```bash
pip install dgl==0.4.1
```

**CUDA 11.1**
```bash
pip install dgl-cu111==0.4.1
```

**CUDA 10.2**
```bash
pip install dgl-cu102==0.4.1
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

## 🧪 Testing

[`test_dataset.py`](test_dataset.py) is a smoke-test that validates every function in `src/dataset.py` using a real `.mat` file from the `data/` folder. No synthetic data — it uses the actual datasets.

```bash
# Default: uses cora.mat (smallest, ~2 sec)
python test_dataset.py

# Test on other datasets
python test_dataset.py --dataset BlogCatalog
python test_dataset.py --dataset ACM
```

**What it tests:**

| Step | Function |
|---|---|
| 1 | `load_mat` — shapes, splits, anomaly labels |
| 2 | `normalize_adj` — symmetric D^{-1/2} A D^{-1/2} normalization |
| 3 | `sparse_mx_to_torch_sparse_tensor` — scipy → torch sparse |
| 4 | `adj_to_dgl_graph` — adjacency → DGL graph |
| 5 | `generate_rwr_subgraph` — RWR subgraph sampling for all nodes |

---

## 🚀 Usage

```bash
python main.py --dataset cora --epochs 100 --lr 0.001
```

> **Note:** Full CLI argument documentation will be added as `main.py` is implemented.

---

## 📊 Supported Datasets

| Dataset | Nodes | Edges | Features | Anomalies |
|---|---|---|---|---|
| Cora | 2,708 | 5,429 | 1,433 | 150 |
| CiteSeer | 3,327 | 4,732 | 3,703 | — |
| Pubmed | 19,717 | 44,338 | 500 | — |
| BlogCatalog | 5,196 | 171,743 | 8,189 | ~5% |
| Flickr | 7,575 | 239,738 | 12,047 | ~5% |
| ACM | 16,484 | 71,980 | 8,337 | ~5% |

Place `.mat` files in the `data/` directory before running.

---

## 📖 Reference Paper

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

## 🗺️ Implementation Roadmap

- [x] `src/dataset.py` — data loading, anomaly injection, subgraph sampling
- [ ] `src/model.py` — GNN encoder, generative decoder, contrastive projector
- [ ] `src/trainer.py` — joint training loop with combined loss
- [ ] `src/evaluator.py` — AUC-ROC / AUC-PR evaluation
- [ ] `main.py` — CLI entry point

---

## 📜 License

This project is for research and educational purposes. Please cite the original paper if you use this code.