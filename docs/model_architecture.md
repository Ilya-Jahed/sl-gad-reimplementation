# `src/model.py` — SL-GAD Model Architecture

> **SL-GAD**: Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection  
> *IEEE Transactions on Knowledge and Data Engineering, 2021*  
> *Forked from GRAND-Lab/CoLA*

This document covers the full model architecture, readout strategies, negative sampling design, the subgraph expansion trick, and data flow with tensor shape annotations.

---

## 📌 Module Overview

`src/model.py` implements the dual-view neural network at the heart of SL-GAD. It is composed of six building blocks:

| Class | Role |
|---|---|
| `GCN` | Graph Convolutional layer — shared **encoder** ($W_{enc}$) and **decoder** ($W_{dec}$) |
| `AvgReadout` | Mean pooling of node embeddings → context summary |
| `MaxReadout` | Max pooling of node embeddings → context summary |
| `MinReadout` | Min pooling of node embeddings → context summary |
| `WSReadout` | Attention-weighted sum pooling using a query vector → context summary |
| `Discriminator` | Bilinear scorer with **in-batch negative sampling** |
| `Model` | Top-level module wiring all components into the full generative + contrastive pipeline |

---

## 🧱 Component Reference

---

### `GCN(in_ft, out_ft, act, bias=True)`

A single GCN layer performing a linear transformation followed by graph neighbourhood aggregation.

**Forward pass:**
$$H = \sigma\!\left(\hat{A} \cdot X \cdot W\right) + b$$

where $\hat{A}$ is the normalized adjacency, $X$ is the node feature matrix, $W$ is the learnable weight, and $\sigma$ is the activation (PReLU by default).

**Parameters**

| Name | Type | Description |
|---|---|---|
| `in_ft` | `int` | Input feature dimension |
| `out_ft` | `int` | Output feature dimension |
| `act` | `str` or callable | `'prelu'` for PReLU; otherwise any `nn.Module` |
| `bias` | `bool` | Whether to add a learnable bias |

**`forward(seq, adj, sparse=False)` — tensor shapes**

| Tensor | Shape | Description |
|---|---|---|
| `seq` | `(B, K+1, in_ft)` | Node feature matrix (K+1 = expanded subgraph, see §Subgraph Expansion) |
| `adj` | `(B, K+1, K+1)` dense or `(N, N)` sparse | Normalized adjacency |
| output | `(B, K+1, out_ft)` | Encoded node embeddings |

- **Dense path** (`sparse=False`): `torch.bmm(adj, seq_fts)` — batch matrix multiply.
- **Sparse path** (`sparse=True`): `torch.spmm(adj, seq_fts)` — used during full-graph evaluation.

**Weight initialization:** Xavier uniform.

---

### Readout Classes

Four readout strategies are available, selected at model construction via the `readout` argument.

#### `AvgReadout` — `forward(seq)`
```python
return torch.mean(seq, 1)    # (B, K-1, n_h) → (B, n_h)
```
Plain mean over all context node embeddings (positions `0` to `K-2`).

#### `MaxReadout` — `forward(seq)`
```python
return torch.max(seq, 1).values    # (B, K-1, n_h) → (B, n_h)
```
Element-wise max over all context node embeddings.

#### `MinReadout` — `forward(seq)`
```python
return torch.min(seq, 1).values    # (B, K-1, n_h) → (B, n_h)
```
Element-wise min over all context node embeddings.

#### `WSReadout` — `forward(seq, query)`

Computes an **attention-weighted** sum of context embeddings using a query vector:

```python
query = query.permute(0, 2, 1)          # (B, n_h, 1)
sim = torch.matmul(seq, query)          # (B, K-1, 1) attention scores
sim = F.softmax(sim, dim=1)             # normalize over context nodes
sim = sim.repeat(1, 1, 64)             # broadcast to (B, K-1, 64)
out = torch.sum(torch.mul(seq, sim), 1) # weighted sum → (B, 64)
```

The `query` is `h[:, -2:-1, :]` — the embedding of the **last context node** — acting as an attention anchor.

> **Note:** `WSReadout` hardcodes `repeat(1, 1, 64)`, so it requires `n_h = 64` (the paper's default embedding dimension) to work correctly.

---

### `Discriminator(n_h, negsamp_round)`

Bilinear scorer with **in-batch negative sampling via cyclic context shift**.

$$s = h^{\top} W_s \, c$$

**Parameters**

| Name | Type | Description |
|---|---|---|
| `n_h` | `int` | Hidden embedding dimension |
| `negsamp_round` | `int` | Number of negative samples per positive (cyclic shifts of `c`) |

**`forward(c, h_pl, s_bias1=None, s_bias2=None)` — tensor shapes**

| Tensor | Shape | Description |
|---|---|---|
| `c` | `(B, n_h)` | Context summary vector |
| `h_pl` | `(B, n_h)` | Target node embedding |
| output | `(B × (1 + negsamp_round), 1)` | Positive + negative logits |

**Negative sampling mechanism:**

```python
scs = []
scs.append(self.f_k(h_pl, c))           # positive pair: target_i vs context_i
c_mi = c
for _ in range(negsamp_round):
    c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)  # cyclic shift
    scs.append(self.f_k(h_pl, c_mi))    # negative pair: target_i vs context_{i+1}
logits = torch.cat(tuple(scs))          # (B*(1+negsamp_round), 1)
```

The cyclic shift creates negatives by misaligning each target node $i$'s embedding with a **different node's context summary** from within the same batch. This avoids any explicit negative sampling overhead since negatives come "for free" from the batch.

**Weight initialization:** Xavier uniform on `nn.Bilinear`.

---

### `Model(n_in, n_h, activation, negsamp_round, readout)`

The full SL-GAD model combining generative attribute reconstruction with multi-view contrastive learning.

**Constructor parameters**

| Name | Type | Description |
|---|---|---|
| `n_in` | `int` | Raw input feature dimension $D$ |
| `n_h` | `int` | Hidden embedding dimension $D'$ (default `64`) |
| `activation` | `str` | Activation type, `'prelu'` |
| `negsamp_round` | `int` | Negative sampling rounds for both discriminators |
| `readout` | `str` | Pooling mode: `'avg'`, `'max'`, `'min'`, `'weighted_sum'` |

**Internal components**

```
gcn_enc  — GCN(n_in → n_h)             Shared encoder for all 4 input views
gcn_dec  — GCN(n_h  → n_in)            Shared decoder for attribute reconstruction
read     — Avg/Max/Min/WSReadout        Context pooling (selected by readout arg)
disc1    — Discriminator(n_h, negsamp)  Contrastive scorer for view 1
disc2    — Discriminator(n_h, negsamp)  Contrastive scorer for view 2
pdist    — PairwiseDistance(p=2)        L2 distance for generative anomaly scoring
```

---

## 🏗️ Subgraph Expansion Trick (Key Design Pattern)

Before data enters the model, the **trainer expands each K-node subgraph to K+1 nodes** by inserting a zero feature row. Understanding this is essential to understanding why the generative module works.

### Feature Expansion

```python
bf = torch.cat((bf[:, :-1, :], zero_row, bf[:, -1:, :]), dim=1)
```

| Position | Index | Features | Adjacency row |
|---|---|---|---|
| Context nodes `0…K-2` | `0` to `K-2` | Original context features | Original context adjacency |
| **Masked slot** | `K-1` **(index `-2`)** | **Zero vector** | **Target node's original adjacency** |
| Target node | `K` **(index `-1`)** | Original target features | Self-loop only (`col[-1] = 1`) |

### Why This Enables Generative Reconstruction

In GCN: $H = \hat{A} \cdot X \cdot W$

For the **masked slot** at index `-2`:
- Its features are zero → it contributes nothing from its own attributes
- Its **adjacency row = target node's original connections** → it **aggregates from context neighbors**
- Result: `h[:, -2, :]` = representation of the target node computed **purely from context**

For the **target slot** at index `-1`:
- Its features are the original target attributes
- Its adjacency row = self-loop only → it encodes only its **own features**, no context

After decoding through `gcn_dec`:
- `f_1[:, -2, :]` = **reconstructed target attributes from context** ← generative output
- `f_1[:, -1, :]` = re-encoded original target features (not used for reconstruction)

This is the core of the generative module: **the masked slot borrows the target's adjacency to reconstruct the target's attributes without ever seeing them.**

---

## 🔄 Data Flow

### Training — `forward(seq1, seq2, seq3, seq4, adj1, adj2)`

The trainer passes:
- `seq1`, `seq2` = preprocessed (row-normalized) features with zero at index `-2`, target at `-1`
- `seq3`, `seq4` = **raw** (unnormalized) features with same structure — used for generative loss

```
seq1 (B, K+1, n_in) ──► gcn_enc(adj1) ──► h_1 (B, K+1, n_h)
seq2 (B, K+1, n_in) ──► gcn_enc(adj2) ──► h_2 (B, K+1, n_h)
seq3 (B, K+1, n_in) ──► gcn_enc(adj1) ──► h_3 ──► gcn_dec(adj1) ──► f_1 (B, K+1, n_in)
seq4 (B, K+1, n_in) ──► gcn_enc(adj2) ──► h_4 ──► gcn_dec(adj2) ──► f_2 (B, K+1, n_in)

── Contrastive branch ──
h_mv_1 = h_1[:, -1, :]           (B, n_h) ← target node embedding, view 1
h_mv_2 = h_2[:, -1, :]           (B, n_h) ← target node embedding, view 2
c1 = read(h_1[:, :-1, :])        (B, n_h) ← context summary, view 1  [positions 0 to K-1]
c2 = read(h_2[:, :-1, :])        (B, n_h) ← context summary, view 2

ret1 = disc1(c1, h_mv_2)  (B*(1+negsamp), 1) ← context_view1 vs target_view2
ret2 = disc2(c2, h_mv_1)  (B*(1+negsamp), 1) ← context_view2 vs target_view1
ret  = cat(ret1, ret2, dim=-1).mean(-1).unsqueeze(-1)  (B*(1+negsamp), 1)
```

**Returns:** `ret`, `f_1`, `f_2`

**Training losses** (computed in trainer):
```python
# Contrastive loss
lbl = [1]*B + [0]*(B*negsamp_round)               # positive + negative labels
loss1 = BCEWithLogitsLoss(ret, lbl)

# Generative loss (MSE between reconstruction and original raw features)
loss2 = 0.5 * (MSE(f_1[:, -2, :], seq3[:, -1, :])  # reconstructed vs original
              + MSE(f_2[:, -2, :], seq4[:, -1, :]))

loss = alpha * loss1 + beta * loss2
```

---

### Inference — `inference(seq1, seq2, seq3, seq4, adj1, adj2)`

Same encoding/decoding, produces two complementary anomaly scores:

```
── Generative anomaly score (attribute deviation) ──
f_1[:, -2, :] = decoded reconstruction at masked slot (context-based)
seq3[:, -1, :] = original raw target features

dist1 = PairwiseDistance(f_1[:, -2, :], seq3[:, -1, :])
dist2 = PairwiseDistance(f_2[:, -2, :], seq4[:, -1, :])
dist  = 0.5 * (dist1 + dist2)     (B,)

── Contrastive anomaly score (structural deviation) ──
ret1 = disc1(c1, h_mv_2)          (B*(1+negsamp), 1)
ret2 = disc2(c2, h_mv_1)          (B*(1+negsamp), 1)
ret  = cat(ret1,ret2).mean(-1).unsqueeze(-1)
```

**Returns:** `ret`, `dist`

**Final anomaly score fusion** (computed in trainer):
```python
# Contrastive component: low agreement = anomalous
ano_score_1 = -(pos_logit - neg_logit)   # per node

# Generative component: high reconstruction error = anomalous
ano_score_2 = dist

# Fused and scaled
ano_score = alpha * normalize(ano_score_1) + beta * normalize(ano_score_2)
```

| Score | High value means | Catches |
|---|---|---|
| `ano_score_1` (contrastive) | Target doesn't fit its structural context | **Structural** anomalies |
| `ano_score_2` (generative) | Target attributes can't be predicted from neighbors | **Attribute** anomalies |

---

## 🧪 Smoke Test

```bash
python test_model.py
```

Covers: `GCN` dense forward, all four readout classes, `Discriminator` with negative sampling (correct output shape `B*(1+negsamp_round)`), `Model.forward()` training pass, `Model.inference()` scoring pass, and `max`/`min` readout mode variants.
