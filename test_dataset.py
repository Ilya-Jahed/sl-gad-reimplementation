"""
test_dataset.py
---------------
Smoke-test for src/dataset.py using a real .mat file from the data/ folder.

Available datasets: cora, citeseer, pubmed, ACM, BlogCatalog, Flickr

Run from the repo root:
    python test_dataset.py
    python test_dataset.py --dataset BlogCatalog
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import (
    dense_to_one_hot,
    normalize_adj,
    sparse_mx_to_torch_sparse_tensor,
    load_mat,
    adj_to_dgl_graph,
    generate_rwr_subgraph,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):
    print(f"  [PASS] {msg}")

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora",
                    choices=["cora", "citeseer", "pubmed", "ACM", "BlogCatalog", "Flickr"],
                    help="Which .mat file to load from data/")
parser.add_argument("--subgraph_size", type=int, default=4,
                    help="Number of nodes per RWR subgraph (including target node)")
args = parser.parse_args()

print(f"\nUsing dataset : {args.dataset}")
print(f"Subgraph size : {args.subgraph_size}")

# ── 1. load_mat ───────────────────────────────────────────────────────────────
section(f"1. load_mat  ({args.dataset}.mat)")

adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano, attr_ano = \
    load_mat(args.dataset)

N = adj.shape[0]

ok(f"adj shape        : {adj.shape}  (sparse, nnz={adj.nnz})")
ok(f"feat shape       : {feat.shape}")
ok(f"labels           : {labels.shape if labels is not None else 'None (no Class key)'}")
ok(f"ano_labels       : {ano_labels.shape}  — anomalies: {ano_labels.sum():.0f} / {N}")
ok(f"str_ano_labels   : {str_ano.shape if str_ano is not None else 'None'}")
ok(f"attr_ano_labels  : {attr_ano.shape if attr_ano is not None else 'None'}")
ok(f"train/val/test   : {len(idx_train)} / {len(idx_val)} / {len(idx_test)}")

assert len(idx_train) + len(idx_val) + len(idx_test) == N

# ── 2. normalize_adj ──────────────────────────────────────────────────────────
section("2. normalize_adj")

norm = normalize_adj(adj)
ok(f"normalized adj shape: {norm.shape}  nnz={norm.nnz}")

# ── 3. sparse_mx_to_torch_sparse_tensor ───────────────────────────────────────
section("3. sparse_mx_to_torch_sparse_tensor")

import torch
t = sparse_mx_to_torch_sparse_tensor(norm)
assert t.is_sparse
ok(f"torch sparse tensor shape: {t.shape}")

# ── 4. adj_to_dgl_graph ───────────────────────────────────────────────────────
section("4. adj_to_dgl_graph")

dgl_graph = adj_to_dgl_graph(adj)
ok(f"DGL graph: {dgl_graph.number_of_nodes()} nodes, {dgl_graph.number_of_edges()} edges")

# ── 5. generate_rwr_subgraph (full graph — subgraph of first 50 nodes causes
#       isolated-node errors because cora edges cross outside that range) ─────
section(f"5. generate_rwr_subgraph  (size={args.subgraph_size}, full graph)")

print("  Running RWR on full graph — may take a few seconds...")
subv = generate_rwr_subgraph(dgl_graph, subgraph_size=args.subgraph_size)

for i, sv in enumerate(subv):
    assert len(sv) == args.subgraph_size, \
        f"Node {i}: got size {len(sv)}, expected {args.subgraph_size}"
    assert sv[-1] == i, \
        f"Node {i}: target node should be last element"

ok(f"Generated {len(subv)} subgraphs, each of size {args.subgraph_size}")
ok(f"Sample — node 0 subgraph: {subv[0]}")
ok(f"Sample — node 1 subgraph: {subv[1]}")

# ── done ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  All tests passed!")
print(f"{'='*55}\n")
