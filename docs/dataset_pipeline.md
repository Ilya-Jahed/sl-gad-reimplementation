# `src/dataset.py` — Data Pipeline Module

> **SL-GAD**: Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection
> *IEEE Transactions on Knowledge and Data Engineering, 2021*

This document provides full technical documentation for the data pipeline module (`src/dataset.py`) and its automated smoke-test suite (`test_dataset.py`).

---

## 📌 Module Overview & Architecture Role

`src/dataset.py` is the **entry point of the entire SL-GAD pipeline**. Before any neural network forward pass occurs, this module is responsible for:

1. Ingesting raw benchmark graph data from `.mat` files
2. Converting graph structures and node features into memory-efficient sparse formats
3. Normalizing the adjacency matrix for stable GCN message-passing
4. Migrating sparse matrices to PyTorch-compatible GPU tensors
5. Sampling fixed-size contextual subgraphs around every node via **Random Walk with Restart (RWR)**

These subgraphs are the direct inputs to both the **Generative Module** (which reconstructs a target node's attributes from its neighborhood) and the **Contrastive Module** (which compares a target node's representation against its context to identify anomalies).

---

## 🔄 Data Processing Pipeline

```
 .mat File (disk)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  load_mat()                                         │
│  ─────────────────────────────────────────────────  │
│  Network   ──►  sp.csr_matrix  (adj)                │
│  Attributes ──►  sp.lil_matrix  (feat)              │
│  Label     ──►  np.ndarray     (ano_labels)         │
│  Class     ──►  one-hot matrix (labels)             │
│  Node idx  ──►  train / val / test split            │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  normalize_adj()                                    │
│  ─────────────────────────────────────────────────  │
│  adj (CSR) ──► Â = D^{-1/2} A D^{-1/2}  (COO)     │
└─────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────────────────────┐
       ▼                                                ▼
┌───────────────────────────┐        ┌──────────────────────────────┐
│  sparse_mx_to_torch       │        │  adj_to_dgl_graph()          │
│  _sparse_tensor()         │        │  ────────────────────        │
│  ─────────────────        │        │  adj (CSR)                   │
│  COO ──► torch            │        │    └► nx.Graph               │
│  .sparse.FloatTensor      │        │         └► dgl.DGLGraph      │
└───────────────────────────┘        └──────────────────────────────┘
       │                                                │
       ▼                                                ▼
  (GCN forward pass)                   ┌──────────────────────────────┐
                                       │  generate_rwr_subgraph()     │
                                       │  ────────────────────        │
                                       │  RWR from every node seed    │
                                       │  ► subv[i]: K-1 context      │
                                       │            + target node i   │
                                       └──────────────────────────────┘
                                                        │
                                                        ▼
                                            (Generative + Contrastive
                                             Module inputs)
```

---

## 🛠 Detailed Function Reference

---

### `dense_to_one_hot(labels_dense, num_classes)`

Converts integer class labels into a binary one-hot matrix.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `labels_dense` | `np.ndarray`, shape `(N,)` | Integer class labels, 0-indexed |
| `num_classes` | `int` | Total number of classes $C$ |

**Returns** — `np.ndarray`, shape `(N, C)`

Uses flat index arithmetic for vectorized assignment:

```python
index_offset = np.arange(num_labels) * num_classes
labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
```

**Example:**
```
labels_dense = [0, 2, 1]  →  [[1, 0, 0],
                               [0, 0, 1],
                               [0, 1, 0]]
```

---

### `load_mat(dataset, train_rate=0.3, val_rate=0.1)`

Loads a `.mat` benchmark graph file and returns all structures needed for training.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `dataset` | `str` | — | Dataset name (e.g. `"cora"`). Resolved as `./data/{dataset}.mat` |
| `train_rate` | `float` | `0.3` | Fraction of nodes for training |
| `val_rate` | `float` | `0.1` | Fraction of nodes for validation |

**Key naming resolution** (handles inconsistencies across `.mat` files):

```python
label   = data['Label']      if 'Label' in data      else data['gnd']
attr    = data['Attributes'] if 'Attributes' in data  else data['X']
network = data['Network']    if 'Network' in data     else data['A']
```

**Returns**

| Variable | Type / Shape | Description |
|---|---|---|
| `adj` | `sp.csr_matrix (N, N)` | Binary adjacency matrix |
| `feat` | `sp.lil_matrix (N, F)` | Node feature matrix |
| `labels` | `np.ndarray (N, C)` | One-hot class labels |
| `idx_train` | `list[int]` | Training node indices |
| `idx_val` | `list[int]` | Validation node indices |
| `idx_test` | `list[int]` | Test node indices |
| `ano_labels` | `np.ndarray (N,)` | Binary anomaly ground truth (1 = anomaly) |
| `str_ano_labels` | `np.ndarray` or `None` | Structural anomaly labels |
| `attr_ano_labels` | `np.ndarray` or `None` | Attribute anomaly labels |

**Split guarantee:** `len(idx_train) + len(idx_val) + len(idx_test) == N`

---

### `normalize_adj(adj)`

Applies symmetric graph Laplacian normalization to the adjacency matrix.

**Parameters** — `adj`: any scipy sparse matrix, shape `(N, N)`

**Returns** — `sp.coo_matrix`, shape `(N, N)`

**Formula:**

$$\hat{A} = D^{-1/2} \, A \, D^{-1/2}$$

where $D$ is the diagonal degree matrix with $D_{ii} = \sum_j A_{ij}$.

**Step-by-step:**

```
1. adj   → COO format
2. rowsum = Σ_j A_{ij}          (degree vector, shape N)
3. d_inv_sqrt = rowsum^{-0.5}   (element-wise power)
4. d_inv_sqrt[isinf] = 0.0      (isolated nodes: 0^{-0.5} = ∞ → clamp to 0)
5. D^{-1/2} = sp.diags(d_inv_sqrt)
6. Â = A · D^{-1/2} · (D^{-1/2})ᵀ
```

Without normalization, repeated GCN aggregation causes node representations to grow unbounded with degree.

---

### `sparse_mx_to_torch_sparse_tensor(sparse_mx)`

Converts a SciPy sparse matrix to a PyTorch sparse COO tensor without materializing a dense intermediate.

**Parameters** — `sparse_mx`: any scipy sparse, shape `(N, N)`

**Returns** — `torch.sparse.FloatTensor`, shape `(N, N)`

```python
sparse_mx = sparse_mx.tocoo().astype(np.float32)

indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # (2, nnz)

values = torch.from_numpy(sparse_mx.data)                        # (nnz,)

return torch.sparse.FloatTensor(indices, values, torch.Size(sparse_mx.shape))
```

**Memory impact (Cora, N=2708, nnz=11606):**

| Format | Storage |
|---|---|
| Dense `float32` | $2708^2 \times 4 \approx$ **29 MB** |
| Sparse COO | $11606 \times 4 \approx$ **45 KB** |
| Reduction | **~640×** |

---

### `adj_to_dgl_graph(adj)`

Converts the SciPy adjacency matrix to a DGL graph to enable DGL's C++ optimized sampling engine.

**Parameters** — `adj`: `sp.csr_matrix (N, N)`

**Returns** — `dgl.DGLGraph`

**Conversion chain:** `sp.csr_matrix → nx.Graph → dgl.DGLGraph`

---

### `generate_rwr_subgraph(dgl_graph, subgraph_size)`

Samples a fixed-size contextual neighborhood for every node using RWR. This is the core sampling function of SL-GAD.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `dgl_graph` | `dgl.DGLGraph` | Full graph |
| `subgraph_size` | `int` | Fixed subgraph size $K$ (including target node) |

**Returns** — `list[list[int]]`, length $N$, each inner list of length exactly $K$

**Structural invariant:** `subv[i][-1] == i` for all $i \in [0, N)$

---

## 💾 Memory Efficiency & Sparse Matrix Strategy

| Format | Used For | Optimized Operation | Memory Layout |
|---|---|---|---|
| **CSR** (Compressed Sparse Row) | `adj` | Row slicing & GCN matrix-vector products | Row pointer + column indices |
| **LIL** (List of Lists) | `feat` | Dynamic row modification (anomaly injection) | List of row-wise lists |
| **COO** (Coordinate) | `normalize_adj` output, bridge format | Efficient construction & format conversion | `(row, col, data)` triplets |
| **Diags** (Diagonal) | Degree matrix $D^{-1/2}$ | $O(N)$ diagonal storage | Single value array |
| **PyTorch Sparse COO** | GCN input tensor | GPU-accelerated sparse-dense matmul | `(indices, values)` on device |

> **Rule of thumb:** CSR → computation · LIL → construction · COO → conversion bridge · PyTorch Sparse → GPU inference

---

## 🎲 Contextual Subgraph Sampling via RWR

### What is RWR?

In **Random Walk with Restart**, a walker starts at a seed node $i$ and at each step either moves to a random neighbor or restarts back to $i$ with restart probability $\alpha$. This produces a neighborhood biased toward the seed, with locality controlled by $\alpha$.

$$P(\text{restart at step } t) = \alpha$$

High $\alpha \approx 1.0$ → strict local (1-hop) sampling.  
Low $\alpha \approx 0.0$ → global exploration.

### Sampling Protocol

**Step 1 — Initial batch walk (all N seeds in parallel)**

```python
traces = dgl.contrib.sampling.random_walk_with_restart(
    dgl_graph,
    seeds              = all_idx,   # all N nodes simultaneously
    restart_prob       = 1.0,       # strict 1-hop neighborhood only
    max_nodes_per_seed = K * 3      # 3× buffer against duplicate visits
)
```

**Step 2 — Deduplication**

```python
subv[i] = torch.unique(torch.cat(trace), sorted=False).tolist()
```

**Step 3 — Fallback for low-degree / isolated nodes**

```python
while len(subv[i]) < K - 1:
    cur_trace = dgl.contrib.sampling.random_walk_with_restart(
        dgl_graph, [i],
        restart_prob       = 0.9,   # relax to allow multi-hop
        max_nodes_per_seed = K * 5
    )
    retry_time += 1
    if retry_time > 10:
        subv[i] = subv[i] * reduced_size  # duplicate to guarantee array size
```

| Condition | `restart_prob` | Step budget | Effect |
|---|---|---|---|
| Normal node | `1.0` | `K × 3` | Strict 1-hop |
| Low-degree (retry) | `0.9` | `K × 5` | Multi-hop exploration |
| Isolated (retry > 10) | — | — | Pad by duplication |

**Step 4 — Truncate & place target node**

```python
subv[i] = subv[i][:K-1]   # exactly K-1 context nodes
subv[i].append(i)          # target node always at index [-1]
```

**Why target is always last:** Both the Generative and Contrastive modules retrieve the target node's embedding as `embeddings[-1]`, enabling consistent positional access across all $N$ subgraphs without bookkeeping overhead.

---

## 🧪 Automated Smoke-Test Suite (`test_dataset.py`)

### Purpose

Validates the complete data pipeline on real `.mat` files before model training begins. Confirms shapes, dtypes, split consistency, sparse properties, and the RWR structural invariant.

### CLI Usage

```bash
# Default: Cora, subgraph_size=4
python test_dataset.py

# Custom dataset and subgraph size
python test_dataset.py --dataset BlogCatalog --subgraph_size 6
python test_dataset.py --dataset ACM
```

**Available datasets:** `cora` · `citeseer` · `pubmed` · `ACM` · `BlogCatalog` · `Flickr`

### Test Sections & Assertions

| Section | Function | Key Assertions |
|---|---|---|
| **1** | `load_mat` | Correct shapes for `adj`, `feat`, `ano_labels`; `len(train)+len(val)+len(test) == N` |
| **2** | `normalize_adj` | Output shape `(N, N)` and `nnz` reported |
| **3** | `sparse_mx_to_torch_sparse_tensor` | `t.is_sparse == True`; shape `(N, N)` |
| **4** | `adj_to_dgl_graph` | Node count == $N$; edge count == `nnz` |
| **5** | `generate_rwr_subgraph` | `len(sv) == K` for all $N$ subgraphs; `sv[-1] == i` for all $i$ |

> **Why full graph for Test 5?** Using a node-index slice (e.g. first 50 nodes) creates a subgraph where seed nodes may have zero edges within that window — Cora's inter-community edges span far beyond contiguous index ranges. DGL raises `no successors from vertex 0` in this case. Running on the full graph avoids this entirely.

### Expected Output (Cora)

```
Using dataset : cora
Subgraph size : 4

=======================================================
  1. load_mat  (cora.mat)
=======================================================
  [PASS] adj shape        : (2708, 2708)  (sparse, nnz=11606)
  [PASS] feat shape       : (2708, 1433)
  [PASS] labels           : (2708, 6)
  [PASS] ano_labels       : (2708,)  — anomalies: 150 / 2708
  [PASS] str_ano_labels   : (2708,)
  [PASS] attr_ano_labels  : (2708,)
  [PASS] train/val/test   : 812 / 270 / 1626

=======================================================
  2. normalize_adj
=======================================================
  [PASS] normalized adj shape: (2708, 2708)  nnz=11606

=======================================================
  3. sparse_mx_to_torch_sparse_tensor
=======================================================
  [PASS] torch sparse tensor shape: torch.Size([2708, 2708])

=======================================================
  4. adj_to_dgl_graph
=======================================================
  [PASS] DGL graph: 2708 nodes, 11606 edges

=======================================================
  5. generate_rwr_subgraph  (size=4, full graph)
=======================================================
  Running RWR on full graph — may take a few seconds...
  [PASS] Generated 2708 subgraphs, each of size 4
  [PASS] Sample — node 0 subgraph: [633, 1862, 2582, 0]
  [PASS] Sample — node 1 subgraph: [451, 2104, 2, 1]

=======================================================
  All tests passed!
=======================================================
```

---

## 🚀 Quick Start

```bash
# 1. Create environment
conda create -n slgad python=3.7 -y && conda activate slgad

# 2. Install PyTorch 1.8.1 (CPU)
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install DGL 0.4.1 (required for RWR sampling API)
pip install dgl==0.4.1

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Run smoke-test
python test_dataset.py
```

**Use in code:**

```python
from src.dataset import (load_mat, normalize_adj,
                         sparse_mx_to_torch_sparse_tensor,
                         adj_to_dgl_graph, generate_rwr_subgraph)

adj, feat, labels, idx_train, idx_val, idx_test, \
    ano_labels, str_ano_labels, attr_ano_labels = load_mat("cora")

norm_adj   = normalize_adj(adj)
adj_tensor = sparse_mx_to_torch_sparse_tensor(norm_adj)   # → GPU ready

dgl_graph  = adj_to_dgl_graph(adj)
subv       = generate_rwr_subgraph(dgl_graph, subgraph_size=4)
# subv[i][-1] == i  ← always holds
```
