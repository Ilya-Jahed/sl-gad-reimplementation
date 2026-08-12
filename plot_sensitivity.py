"""
plot_sensitivity.py
--------------------
Hyperparameter sensitivity analysis for SL-GAD.

Reproduces the standard 2x2 AUC-sensitivity figure:
    (a) Evaluation rounds  vs AUC
    (b) Subgraph size      vs AUC
    (c) Hidden dimension   vs AUC
    (d) Negative ratio     vs AUC

For every dataset passed via --datasets, the script trains (once per
config) and evaluates the SL-GAD model across a range of values for each
hyperparameter, caches the resulting AUCs to a JSON file, and plots all
datasets together on one figure per panel (or one combined 2x2 figure).

This does NOT modify main.py / src/trainer.py / src/evaluator.py — it
reuses those classes directly, so it drops straight into the existing
repo layout (place it next to main.py).

Usage
-----
    # full run (slow — trains a fresh model for every point on every
    # dataset for the subgraph-size / hidden-dim / negative-ratio panels)
    python plot_sensitivity.py --datasets cora citeseer BlogCatalog

    # fast smoke-test (caps epochs/patience so you can check the pipeline
    # and plotting code work before committing to a full run)
    python plot_sensitivity.py --datasets cora --quick

    # only the cheap panel (no retraining, just re-evaluates with more
    # inference rounds using one trained model per dataset)
    python plot_sensitivity.py --datasets cora citeseer --panels eval_rounds

    # re-plot from a previous run's cache without recomputing anything
    python plot_sensitivity.py --datasets cora citeseer --plot_only \
        --cache results/sensitivity_cache.json
"""

import os
import json
import argparse

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import (
    load_mat,
    preprocess_features,
    normalize_adj,
    adj_to_dgl_graph,
)
from src.model import Model
from src.trainer import Trainer
from src.evaluator import Evaluator


# ─────────────────────────────────────────────────────────────
# Reproducibility (duplicated from main.py — kept self-contained
# so this script never triggers main.py's module-level argparse)
# ─────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    import dgl
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_defaults(dataset):
    """lr / num_epoch defaults, mirrors main.py::apply_dataset_defaults."""
    if dataset in ['cora', 'citeseer', 'pubmed', 'Flickr', 'ACM']:
        lr = 1e-3
    elif dataset == 'BlogCatalog':
        lr = 3e-3
    else:
        lr = 1e-3

    if dataset in ['cora', 'citeseer', 'pubmed']:
        num_epoch = 100
    elif dataset in ['BlogCatalog', 'Flickr', 'ACM']:
        num_epoch = 400
    else:
        num_epoch = 200

    return lr, num_epoch


BASE_CFG = dict(
    weight_decay=0.0,
    embedding_dim=64,
    patience=400,
    batch_size=300,
    subgraph_size=4,
    readout='avg',
    auc_test_rounds=256,
    negsamp_ratio=1,
    alpha=1.0,
    beta=0.6,
)


# ─────────────────────────────────────────────────────────────
# Data loading (same steps as main.py, factored into a function)
# ─────────────────────────────────────────────────────────────

def load_dataset(name, device):
    adj, features, labels, idx_train, idx_val, idx_test, \
        ano_label, str_ano_label, attr_ano_label = load_mat(name)

    raw_features = features.todense()
    features = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)

    return adj, features, raw_features, dgl_graph, nb_nodes, ft_size, ano_label


# ─────────────────────────────────────────────────────────────
# One train+eval run for a given config
# ─────────────────────────────────────────────────────────────

def train_and_eval(dataset, device, seed=1, quick=False, run_tag='run', **overrides):
    cfg = dict(BASE_CFG)
    lr, num_epoch = dataset_defaults(dataset)
    cfg['lr'] = lr
    cfg['num_epoch'] = num_epoch
    cfg.update(overrides)

    if quick:
        cfg['num_epoch'] = min(cfg['num_epoch'], 15)
        cfg['patience'] = min(cfg['patience'], 15)

    set_seed(seed)
    data = load_dataset(dataset, device)
    adj, features, raw_features, dgl_graph, nb_nodes, ft_size, ano_label = data

    model = Model(ft_size, cfg['embedding_dim'], 'prelu',
                  cfg['negsamp_ratio'], cfg['readout']).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    b_xent = nn.BCEWithLogitsLoss(
        reduction='none',
        pos_weight=torch.tensor([cfg['negsamp_ratio']]).to(device),
    )
    mse_loss = nn.MSELoss(reduction='mean')

    trainer = Trainer(
        model=model, optimiser=optimiser, b_xent=b_xent, mse_loss=mse_loss,
        device=device, alpha=cfg['alpha'], beta=cfg['beta'],
        negsamp_ratio=cfg['negsamp_ratio'], batch_size=cfg['batch_size'],
        num_epoch=cfg['num_epoch'], patience=cfg['patience'],
        subgraph_size=cfg['subgraph_size'],
    )

    expid = f"sens_{dataset}_{run_tag}"
    trainer.train(adj, features, raw_features, dgl_graph, nb_nodes, ft_size, expid)
    model.load_state_dict(torch.load(f'checkpoints/exp_{expid}.pkl'))

    evaluator = Evaluator(
        model=model, device=device, alpha=cfg['alpha'], beta=cfg['beta'],
        negsamp_ratio=cfg['negsamp_ratio'], batch_size=cfg['batch_size'],
        subgraph_size=cfg['subgraph_size'], auc_test_rounds=cfg['auc_test_rounds'],
    )
    auc = evaluator.evaluate(adj, features, raw_features, dgl_graph,
                              nb_nodes, ft_size, ano_label)
    return auc, model, data, cfg


# ─────────────────────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────────────────────

def sweep_eval_rounds(dataset, device, values, seed=1, quick=False):
    """Cheap: train ONE model, then re-run evaluate() with different
    auc_test_rounds — no retraining needed."""
    auc, model, data, cfg = train_and_eval(dataset, device, seed=seed,
                                            quick=quick, run_tag='eval_rounds')
    adj, features, raw_features, dgl_graph, nb_nodes, ft_size, ano_label = data
    results = []
    for r in values:
        evaluator = Evaluator(
            model=model, device=device, alpha=cfg['alpha'], beta=cfg['beta'],
            negsamp_ratio=cfg['negsamp_ratio'], batch_size=cfg['batch_size'],
            subgraph_size=cfg['subgraph_size'], auc_test_rounds=r,
        )
        a = evaluator.evaluate(adj, features, raw_features, dgl_graph,
                                nb_nodes, ft_size, ano_label)
        results.append(a)
    return results


def sweep_param(dataset, device, values, param_name, seed=1, quick=False):
    """Expensive: retrain from scratch for every value (subgraph_size,
    embedding_dim, negsamp_ratio all change the model/data shape)."""
    results = []
    for v in values:
        overrides = {param_name: v}
        auc, _, _, _ = train_and_eval(
            dataset, device, seed=seed, quick=quick,
            run_tag=f'{param_name}_{v}', **overrides,
        )
        results.append(auc)
    return results


PANEL_SPECS = {
    'eval_rounds':   dict(values=[1, 5, 10, 20, 40, 80, 160, 320],
                           xlabel='Evaluation Rounds', param=None),
    'subgraph_size': dict(values=[2, 4, 6, 8, 10, 12, 14],
                           xlabel='Subgraph Size', param='subgraph_size'),
    'hidden_dim':    dict(values=[2, 4, 8, 16, 32, 64, 128, 256],
                           xlabel='Hidden Dimension', param='embedding_dim'),
    'negative_ratio': dict(values=[1, 2, 4, 8, 16, 32, 64, 128],
                            xlabel='Negative Ratio', param='negsamp_ratio'),
}

DATASET_STYLE = {
    'BlogCatalog': dict(color='tab:red', marker='o'),
    'Flickr':      dict(color='tab:blue', marker='v'),
    'ACM':         dict(color='tab:cyan', marker='+'),
    'cora':        dict(color='tab:green', marker='s'),
    'Cora':        dict(color='tab:green', marker='s'),
    'citeseer':    dict(color='yellowgreen', marker='*'),
    'Citeseer':    dict(color='yellowgreen', marker='*'),
    'pubmed':      dict(color='tab:purple', marker='^'),
    'Pubmed':      dict(color='tab:purple', marker='^'),
}
DEFAULT_STYLE_CYCLE = ['tab:orange', 'tab:brown', 'tab:pink', 'gray']


def run_all(datasets, panels, device, seed, quick):
    """results[panel][dataset] = list of AUCs matching PANEL_SPECS[panel]['values']"""
    results = {p: {} for p in panels}
    for dataset in datasets:
        print(f'\n=== Dataset: {dataset} ===', flush=True)
        for panel in panels:
            spec = PANEL_SPECS[panel]
            print(f'-- panel: {panel} --', flush=True)
            if panel == 'eval_rounds':
                ys = sweep_eval_rounds(dataset, device, spec['values'],
                                        seed=seed, quick=quick)
            else:
                ys = sweep_param(dataset, device, spec['values'],
                                  spec['param'], seed=seed, quick=quick)
            results[panel][dataset] = ys
            print(f'{dataset} / {panel}: {ys}', flush=True)
    return results


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_panel(ax, values, results_by_dataset, xlabel, title):
    extra_colors = iter(DEFAULT_STYLE_CYCLE)
    for name, ys in results_by_dataset.items():
        style = DATASET_STYLE.get(name)
        if style is None:
            style = dict(color=next(extra_colors, 'black'), marker='x')
        ax.plot(values, ys, label=name, linewidth=1.5, markersize=6, **style)
    ax.set_xscale('log', base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('AUC')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


PANEL_TITLES = {
    'eval_rounds':    '(a) Evaluation rounds versus AUC values',
    'subgraph_size':  '(b) Subgraph size versus AUC values',
    'hidden_dim':     '(c) Hidden dimension versus AUC values',
    'negative_ratio': '(d) Negative ratio versus AUC values',
}


def plot_all(results, panels, out_path):
    n = len(panels)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, panel in zip(axes, panels):
        spec = PANEL_SPECS[panel]
        plot_panel(ax, spec['values'], results[panel], spec['xlabel'],
                   PANEL_TITLES[panel])
    for ax in axes[len(panels):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'\nSaved figure to {out_path}', flush=True)


# ─────────────────────────────────────────────────────────────
# Cache helpers (so you don't have to retrain to re-plot)
# ─────────────────────────────────────────────────────────────

def save_cache(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_cache(path):
    with open(path, 'r') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SL-GAD sensitivity plots')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                         help='Dataset names matching data/<name>.mat')
    parser.add_argument('--panels', type=str, nargs='+',
                         default=list(PANEL_SPECS.keys()),
                         choices=list(PANEL_SPECS.keys()),
                         help='Which panels to compute/plot')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--quick', action='store_true',
                         help='Cap epochs/patience for a fast smoke test')
    parser.add_argument('--cache', type=str,
                         default='results/sensitivity_cache.json',
                         help='Where to save/load computed AUCs')
    parser.add_argument('--out', type=str,
                         default='results/sensitivity_plot.png',
                         help='Output figure path')
    parser.add_argument('--plot_only', action='store_true',
                         help='Skip training, just re-plot from --cache')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.plot_only:
        results = load_cache(args.cache)
    else:
        results = run_all(args.datasets, args.panels, device, args.seed, args.quick)
        save_cache(results, args.cache)

    plot_all(results, args.panels, args.out)


if __name__ == '__main__':
    main()