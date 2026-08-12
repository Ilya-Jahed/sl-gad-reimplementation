# Training, Evaluation & CLI Pipeline

> **SL-GAD**: Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection  
> *IEEE Transactions on Knowledge and Data Engineering, 2021*

This document covers the three files that together form the **complete SL-GAD pipeline**:

| File | Responsibility |
|---|---|
| [`src/trainer.py`](../src/trainer.py) | Subgraph batch construction, joint loss computation, early stopping |
| [`src/evaluator.py`](../src/evaluator.py) | Multi-round stochastic inference, anomaly score fusion, AUC evaluation |
| [`main.py`](../main.py) | CLI entry point — wires data, model, trainer, and evaluator together |

---

## 📌 Design Principles

All three files are direct modular extractions from the original monolithic `run.py` from GRAND-Lab/CoLA. The logic is **identical to the source** — no algorithmic changes have been made. The modularisation improves:

- **Readability**: each concern is isolated to a focused class or function.
- **Testability**: the `Trainer` and `Evaluator` can be instantiated independently.
- **Reusability**: `build_batch()` is shared between training and evaluation without duplication.

---

## 🏗️ The Subgraph Expansion Trick

Before describing any of the three files, it is essential to understand **how the data batch is constructed**, since this design choice underpins everything.

### Problem
The Generative module must reconstruct a target node's attributes **without ever seeing them** — it must infer them purely from the context subgraph.

### Solution — $K \to K+1$ Node Expansion

The trainer inserts a **zero-feature row** into each subgraph at position $-2$, pushing the original target node to position $-1$:

```
Before expansion (K nodes):
  [ ctx_0, ctx_1, ..., ctx_{K-2}, TARGET ]
    
After expansion (K+1 nodes):
  [ ctx_0, ctx_1, ..., ctx_{K-2},  ZERO ,  TARGET ]
                                    -2        -1
```

The adjacency matrix is expanded to match, but critically:
- **Position `-2`** (the zero row): inherits the **target node's original adjacency row** → GCN aggregates context neighbours → output = context-based reconstruction of target
- **Position `-1`** (the target slot): has only a self-loop → GCN encodes only its own features

After the GCN encoder + decoder:
```
f[:, -2, :] = reconstructed target attributes (from context only)
f[:, -1, :] = re-encoded original target features (not used in loss)
```

The generative MSE loss is therefore:
$$\mathcal{L}_{\text{gen}} = \text{MSE}\bigl(f[\text{:,-2,:}],\ \text{raw\_feat}[\text{:,-1,:}]\bigr)$$

This function is implemented once in `build_batch()` and reused in both training and evaluation.

---

## 📄 `src/trainer.py`

### `build_batch()` — Shared Batch Builder

```python
def build_batch(idx, subgraphs_1, subgraphs_2,
                adj, features, raw_features,
                ft_size, subgraph_size, device)
```

A **module-level function** (not a class method) so it can be imported and reused by `Evaluator` without duplicating logic.

**What it does, step by step:**

1. **Slice subgraph adjacency**: for each node index `i`, slice the full $N \times N$ adjacency to get the $K \times K$ subgraph adjacency for both views.
2. **Slice features**: slice both normalised and raw feature matrices to get $K \times F$ tensors.
3. **Expand adjacency** $(K \times K) \to (K+1 \times K+1)$:
   - Concatenate a zero row at the bottom (dim=1) — the new masked slot has no outgoing edges.
   - Concatenate a zero column at the right (dim=2), with `col[-1] = 1` — the target slot has a self-loop only.
4. **Expand features** $(K \times F) \to (K+1 \times F)$:
   - Insert a zero row between the context nodes and the target: `[ctx_nodes | ZERO | target]`

**Tensor shapes through `build_batch`:**

| Tensor | Input shape | Output shape |
|---|---|---|
| `ba1`, `ba2` | `(B, K, K)` | `(B, K+1, K+1)` |
| `bf1`, `bf2` | `(B, K, F)` | `(B, K+1, F)` |
| `raw_bf1`, `raw_bf2` | `(B, K, F)` | `(B, K+1, F)` |

---

### `Trainer` Class

```python
Trainer(model, optimiser, b_xent, mse_loss, device,
        alpha, beta, negsamp_ratio,
        batch_size, num_epoch, patience, subgraph_size)
```

#### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `nn.Module` | The SL-GAD `Model` instance |
| `optimiser` | `Optimizer` | Adam optimizer |
| `b_xent` | `BCEWithLogitsLoss` | Contrastive loss (positive-weighted) |
| `mse_loss` | `MSELoss` | Generative reconstruction loss |
| `device` | `torch.device` | Target device (CPU or CUDA) |
| `alpha` | `float` | Contrastive loss weight (default `1.0`) |
| `beta` | `float` | Generative loss weight (default `0.6`) |
| `negsamp_ratio` | `int` | Negative samples per positive (default `1`) |
| `batch_size` | `int` | Nodes processed per batch (default `300`) |
| `num_epoch` | `int` | Max training epochs |
| `patience` | `int` | Early stopping patience in epochs |
| `subgraph_size` | `int` | $K$ — subgraph size before expansion |

#### `Trainer.train()` — Training Loop

```python
best_epoch = trainer.train(adj, features, raw_features, dgl_graph, nb_nodes, ft_size, expid)
```

**Epoch-level logic:**
1. Shuffle all node indices.
2. Sample **two independent RWR subgraph views** for the entire graph via `generate_rwr_subgraph()`.
3. Iterate over mini-batches, calling `build_batch()` for each.

**Batch-level logic:**

```
lbl = [1, 1, ..., 1,  0, 0, ..., 0]   ← B positives + B*negsamp_ratio negatives
       ←── B ───→   ←─── B*negsamp ──→

logits, f_1, f_2 = model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)

loss1 = mean(BCEWithLogitsLoss(logits, lbl))          ← contrastive
loss2 = 0.5 * (MSE(f_1[:,-2,:], raw_bf1[:,-1,:])
             + MSE(f_2[:,-2,:], raw_bf2[:,-1,:]))     ← generative
loss  = alpha * loss1 + beta * loss2
```

**Epoch loss accumulation:**

The source uses a numerically stable weighted mean:
$$\mathcal{L}_{\text{epoch}} = \frac{\text{total\_loss} \times \text{batch\_size} + \text{last\_batch\_loss} \times \text{last\_batch\_size}}{N}$$

**Early stopping and checkpointing:**
- If the epoch loss improves → save checkpoint to `checkpoints/exp_{expid}.pkl`.
- If no improvement for `patience` epochs → stop training and return `best_t`.

**Returns:** `best_t` — the epoch index of the best saved checkpoint.

---

## 📄 `src/evaluator.py`

### `Evaluator` Class

```python
Evaluator(model, device, alpha, beta, negsamp_ratio,
          batch_size, subgraph_size, auc_test_rounds)
```

#### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `nn.Module` | Trained model (loaded from checkpoint) |
| `device` | `torch.device` | |
| `alpha` | `float` | Contrastive score weight |
| `beta` | `float` | Generative score weight |
| `negsamp_ratio` | `int` | Must match value used during training |
| `batch_size` | `int` | |
| `subgraph_size` | `int` | $K$ (before expansion) |
| `auc_test_rounds` | `int` | Number of inference rounds (default `256`) |

---

### `_compute_ano_score()` — Score Fusion

```python
ano_score = evaluator._compute_ano_score(logits, dist, cur_batch_size)
```

For a single batch, computes the fused anomaly score under three operating modes:

#### Mode 1: Both modules active (`alpha != 0` and `beta != 0`)

$$\text{score}_1 = -\bigl(\sigma(\text{logits}_{\text{pos}}) - \sigma(\text{logits}_{\text{neg}})\bigr)$$
$$\text{score}_2 = \text{dist}$$
$$\text{ano\_score} = \alpha \cdot \text{MinMax}(\text{score}_1) + \beta \cdot \text{MinMax}(\text{score}_2)$$

Both components are independently scaled to $[0, 1]$ before fusion to ensure neither dominates due to scale differences.

#### Mode 2: Contrastive only (`beta == 0`)

$$\text{ano\_score} = -\bigl(\sigma(\text{logits}_{\text{pos}}) - \sigma(\text{logits}_{\text{neg}})\bigr)$$

#### Mode 3: Generative only (`alpha == 0`)

$$\text{ano\_score} = \text{dist}$$

**Handling `negsamp_ratio > 1`:**

When multiple negative samples exist per positive:
```python
neg_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
ano_score_1 = -(pos_score - neg_score)
```
The negatives are reshaped and averaged before subtraction.

---

### `Evaluator.evaluate()` — Multi-round AUC

```python
auc = evaluator.evaluate(adj, features, raw_features, dgl_graph, nb_nodes, ft_size, ano_label)
```

**Why multiple rounds?**

RWR subgraph sampling is stochastic. A single inference pass produces a noisy anomaly score per node. Running `auc_test_rounds` rounds with **independent random subgraph views** and averaging the resulting scores significantly reduces variance and produces a stable final ranking.

**Algorithm:**
```
multi_round_ano_score = zeros(auc_test_rounds, nb_nodes)

for round in range(auc_test_rounds):
    sample fresh subgraphs_1, subgraphs_2
    for each batch:
        logits, dist = model.inference(...)
        ano_score[batch_idx] = _compute_ano_score(...)
    multi_round_ano_score[round, idx] = ano_score

ano_score_final = mean(multi_round_ano_score, axis=0)   ← per-node mean across rounds
auc = roc_auc_score(ano_label, ano_score_final)
```

**Returns:** `auc` — ROC-AUC score (higher is better; `1.0` = perfect separation).

---

## 📄 `main.py`

The CLI entry point that connects all modules.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--expid` | *(required)* | Experiment ID for checkpoint naming |
| `--dataset` | `BlogCatalog` | Dataset name (must match `data/<name>.mat`) |
| `--device` | `cuda:0` | Target device; falls back to CPU automatically |
| `--runs` | `1` | Number of independent runs with different seeds |
| `--embedding_dim` | `64` | Hidden embedding dimension $n_h$ |
| `--lr` | auto | Learning rate (auto-set per dataset) |
| `--num_epoch` | auto | Max epochs (auto-set per dataset) |
| `--patience` | `400` | Early stopping patience |
| `--batch_size` | `300` | Nodes per mini-batch |
| `--subgraph_size` | `4` | $K$ — RWR subgraph size before expansion |
| `--readout` | `avg` | Pooling mode: `avg`, `max`, `min`, `weighted_sum` |
| `--negsamp_ratio` | `1` | Negatives per positive in Discriminator |
| `--alpha` | `1.0` | Contrastive loss/score weight |
| `--beta` | `0.6` | Generative loss/score weight |
| `--auc_test_rounds` | `256` | Inference rounds for AUC estimation |

### Dataset-Specific Defaults

| Dataset | `lr` | `num_epoch` |
|---|---|---|
| `cora`, `citeseer`, `pubmed` | `1e-3` | `100` |
| `BlogCatalog`, `Flickr`, `ACM` | `3e-3` / `1e-3` | `400` |

### Data Preprocessing Pipeline

```
load_mat()          → adj (CSR), features (LIL sparse), ano_labels
features.todense()  → raw_features  (unnormalized — for generative loss)
preprocess_features(features) → features (row-normalized — for GCN input)
adj_to_dgl_graph(adj) → dgl_graph  (for RWR sampling)
normalize_adj(adj)  → adj (D^{-1/2} A D^{-1/2})
adj + I             → adj (with self-loops)

# Move to device with batch dimension [np.newaxis]
features     → torch.FloatTensor (1, N, F)
raw_features → torch.FloatTensor (1, N, F)
adj          → torch.FloatTensor (1, N, N)
```

### Run Flow

```
for each run:
    set_seed(seed)
    Model → Trainer → trainer.train() → best_epoch
    load checkpoint from checkpoints/exp_{expid}.pkl
    Evaluator → evaluator.evaluate() → auc
    
print FINAL TESTING AUC: mean(all_auc)
```

---

## 🚀 Usage Examples

```bash
# Minimal run on Cora (100 epochs, auto lr)
python main.py --expid 1 --dataset cora

# BlogCatalog with CUDA
python main.py --expid 2 --dataset BlogCatalog --device cuda:0

# 3 runs for statistical reliability
python main.py --expid 3 --dataset ACM --runs 3

# Contrastive only (beta=0 disables generative score at eval)
python main.py --expid 4 --dataset cora --alpha 1.0 --beta 0.0

# Faster evaluation for debugging
python main.py --expid 5 --dataset cora --auc_test_rounds 10
```

---

## 📊 Verified Results

| Dataset | AUC (This Implementation) | AUC (Paper) |
|---|---|---|
| Cora | **0.9167** | ~0.90–0.93 |

> Result verified with: `--expid 1 --dataset cora --runs 1 --auc_test_rounds 256`

---

## 🔗 Related Documentation

- [Dataset Pipeline](dataset_pipeline.md) — `src/dataset.py` data loading and RWR subgraph sampling
- [Model Architecture](model_architecture.md) — `src/model.py` GCN, Discriminator, and readout modules
