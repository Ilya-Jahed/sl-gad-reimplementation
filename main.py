"""
main.py
-------
CLI entry point for SL-GAD: Generative and Contrastive Self-Supervised
Learning for Graph Anomaly Detection.

Usage:
    python main.py --expid 1 --dataset cora
    python main.py --expid 2 --dataset BlogCatalog --runs 3 --device cuda:0

Forked and modularised from GRAND-Lab/CoLA run.py
"""

import os
import random
import argparse

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import dgl

from src.dataset import (
    load_mat,
    preprocess_features,
    normalize_adj,
    adj_to_dgl_graph,
)
from src.model import Model
from src.trainer import Trainer
from src.evaluator import Evaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='SL-GAD')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--expid',           type=int,   required=True,
                    help='Experiment ID (used for checkpoint filename)')
parser.add_argument('--device',          type=str,   default='cuda:0')
parser.add_argument('--dataset',         type=str,   default='BlogCatalog',
                    help='Dataset name matching data/<dataset>.mat')
parser.add_argument('--lr',              type=float, default=None,
                    help='Learning rate (auto-set per dataset if omitted)')
parser.add_argument('--weight_decay',    type=float, default=0.0)
parser.add_argument('--runs',            type=int,   default=1,
                    help='Number of independent training runs')
parser.add_argument('--embedding_dim',   type=int,   default=64,
                    help='Hidden embedding dimension n_h')
parser.add_argument('--patience',        type=int,   default=400,
                    help='Early stopping patience (epochs)')
parser.add_argument('--num_epoch',       type=int,   default=None,
                    help='Max training epochs (auto-set per dataset if omitted)')
parser.add_argument('--drop_prob',       type=float, default=0.0)
parser.add_argument('--batch_size',      type=int,   default=300)
parser.add_argument('--subgraph_size',   type=int,   default=4,
                    help='K: nodes per RWR subgraph (before expansion)')
parser.add_argument('--readout',         type=str,   default='avg',
                    choices=['avg', 'max', 'min', 'weighted_sum'])
parser.add_argument('--auc_test_rounds', type=int,   default=256,
                    help='Stochastic inference rounds for AUC estimation')
parser.add_argument('--negsamp_ratio',   type=int,   default=1,
                    help='Negatives per positive in Discriminator')
parser.add_argument('--alpha',           type=float, default=1.0,
                    help='Contrastive loss / score weight')
parser.add_argument('--beta',            type=float, default=0.6,
                    help='Generative loss / score weight')

args = parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Dataset-specific defaults (from paper hyperparameters)
# ─────────────────────────────────────────────────────────────

def apply_dataset_defaults(args):
    if args.lr is None:
        if args.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr', 'ACM']:
            args.lr = 1e-3
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3
        else:
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
            args.num_epoch = 400
        else:
            args.num_epoch = 200

    return args


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed):
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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = apply_dataset_defaults(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f'Dataset: {args.dataset}', flush=True)
    print(f'Device : {device}', flush=True)

    # ── Data loading ─────────────────────────────────────────
    adj, features, labels, idx_train, idx_val, idx_test, \
        ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    # raw_features: unnormalized — used by generative MSE loss
    raw_features = features.todense()
    # features: row-normalized — used as GCN encoder input
    features = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size  = features.shape[1]

    # Symmetric normalization + self-loops for GCN
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    # Move to device, add batch dimension
    features     = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj          = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels       = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train    = torch.LongTensor(idx_train).to(device)
    idx_val      = torch.LongTensor(idx_val).to(device)
    idx_test     = torch.LongTensor(idx_test).to(device)

    # ── Multi-run loop ────────────────────────────────────────
    seeds   = [i + 1 for i in range(args.runs)]
    all_auc = []

    for run in range(args.runs):
        seed = seeds[run]
        print(f'\n# Run: {run}  Seed: {seed}', flush=True)
        set_seed(seed)

        # ── Model, optimiser, losses ──────────────────────────
        model = Model(
            ft_size,
            args.embedding_dim,
            'prelu',
            args.negsamp_ratio,
            args.readout,
        ).to(device)

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        b_xent = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.tensor([args.negsamp_ratio]).to(device),
        )
        mse_loss = nn.MSELoss(reduction='mean')

        # ── Training ──────────────────────────────────────────
        trainer = Trainer(
            model=model,
            optimiser=optimiser,
            b_xent=b_xent,
            mse_loss=mse_loss,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            negsamp_ratio=args.negsamp_ratio,
            batch_size=args.batch_size,
            num_epoch=args.num_epoch,
            patience=args.patience,
            subgraph_size=args.subgraph_size,
        )

        best_epoch = trainer.train(
            adj=adj,
            features=features,
            raw_features=raw_features,
            dgl_graph=dgl_graph,
            nb_nodes=nb_nodes,
            ft_size=ft_size,
            expid=args.expid,
        )

        # ── Load best checkpoint ──────────────────────────────
        print(f'Loading epoch {best_epoch} checkpoint', flush=True)
        model.load_state_dict(torch.load(f'checkpoints/exp_{args.expid}.pkl'))

        # ── Evaluation ────────────────────────────────────────
        evaluator = Evaluator(
            model=model,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            negsamp_ratio=args.negsamp_ratio,
            batch_size=args.batch_size,
            subgraph_size=args.subgraph_size,
            auc_test_rounds=args.auc_test_rounds,
        )

        auc = evaluator.evaluate(
            adj=adj,
            features=features,
            raw_features=raw_features,
            dgl_graph=dgl_graph,
            nb_nodes=nb_nodes,
            ft_size=ft_size,
            ano_label=ano_label,
        )

        all_auc.append(auc)
        print(f'Testing AUC: {auc:.4f}', flush=True)

    # ── Final summary ─────────────────────────────────────────
    print('\n' + '=' * 40)
    print('Per-run AUC:', all_auc)
    print(f'FINAL TESTING AUC: {np.mean(all_auc):.4f}')
    print('=' * 40)
