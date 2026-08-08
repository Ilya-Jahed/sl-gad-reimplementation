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
├── data/                  # Raw & processed datasets (Cora, BlogCatalog, …)
├── src/
│   ├── __init__.py
│   ├── dataset.py         # Data loading & subgraph sampling
│   ├── model.py           # Generative + Contrastive model architectures
│   ├── trainer.py         # Training loop & loss computation
│   └── evaluator.py       # AUC-ROC scoring & evaluation utilities
├── main.py                # Entry point – run the full pipeline
├── requirements.txt       # Python package dependencies
└── README.md
```

---

## ⚙️ Environment Setup (Conda)

### 1. Create and activate a new conda environment

```bash
conda create -n slgad python=3.10 -y
conda activate slgad
```

### 2. Install PyTorch (choose the command matching your hardware)

**CUDA 11.8**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**CUDA 12.1**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU only**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 🚀 Usage

```bash
python main.py --dataset cora --epochs 100 --lr 0.001
```

> **Note:** Full CLI argument documentation will be added as `main.py` is implemented.

---

## 📊 Supported Datasets

| Dataset | Nodes | Edges | Features | Anomaly ratio |
|---|---|---|---|---|
| Cora | 2,708 | 5,429 | 1,433 | ~5 % |
| BlogCatalog | 5,196 | 171,743 | 8,189 | ~5 % |
| Flickr | 7,575 | 239,738 | 12,047 | ~5 % |
| ACM | 16,484 | 71,980 | 8,337 | ~5 % |

---

## 📖 Reference Paper

```bibtex
@inproceedings{zheng2021generative,
  title     = {Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection},
  author    = {Zheng, Yu and Jin, Ming and Liu, Yixin and Chi, Lianhua and Phan, Khoa T. and Chen, Yi-Ping Phoebe},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  year      = {2021},
  doi       = {10.1109/TKDE.2021.3119326}
}
```

---

## 🗺️ Implementation Roadmap

- [ ] `src/dataset.py` — data loading, anomaly injection, subgraph sampling
- [ ] `src/model.py` — GNN encoder, generative decoder, contrastive projector
- [ ] `src/trainer.py` — joint training loop with combined loss
- [ ] `src/evaluator.py` — AUC-ROC / AUC-PR evaluation
- [ ] `main.py` — CLI entry point

---

## 📜 License

This project is for research and educational purposes. Please cite the original paper if you use this code.