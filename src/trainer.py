"""
src/trainer.py
--------------
Training loop for SL-GAD.

Handles:
  - Subgraph batch construction (the K -> K+1 node expansion trick)
  - Joint contrastive (BCE) + generative (MSE) loss computation
  - Early stopping with model checkpointing

Forked and modularised from GRAND-Lab/CoLA run.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from src.dataset import generate_rwr_subgraph


# ─────────────────────────────────────────────────────────────
# Shared batch builder (used by both Trainer and Evaluator)
# ─────────────────────────────────────────────────────────────

def build_batch(idx, subgraphs_1, subgraphs_2,
                adj, features, raw_features,
                ft_size, subgraph_size, device):
    """
    Expand a list of node indices into a padded K+1-node batch.

    The original K-node subgraph (K-1 context + 1 target) is expanded to
    K+1 nodes by inserting a zero-feature row at position -2, pushing the
    target to position -1. The adjacency is expanded so that position -2
    inherits the target's original adjacency row, enabling the GCN to
    reconstruct the target purely from its context neighbours.

    Parameters
    ----------
    idx           : list[int]        node indices in this batch
    subgraphs_1   : list[list[int]]  RWR subgraphs for view 1
    subgraphs_2   : list[list[int]]  RWR subgraphs for view 2
    adj           : torch.FloatTensor  (1, N, N) normalised adjacency on device
    features      : torch.FloatTensor  (1, N, F) row-normalised features
    raw_features  : torch.FloatTensor  (1, N, F) raw (unnormalised) features
    ft_size       : int   feature dimension F
    subgraph_size : int   K (nodes per subgraph before expansion)
    device        : torch.device

    Returns
    -------
    ba1, ba2          : (B, K+1, K+1) expanded adjacency tensors
    bf1, bf2          : (B, K+1, F)   normalised feature tensors
    raw_bf1, raw_bf2  : (B, K+1, F)   raw feature tensors
    """
    cur_batch_size = len(idx)

    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
    added_adj_zero_col[:, -1, :] = 1.          # self-loop for the target slot
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

    ba1, ba2 = [], []
    bf1, bf2 = [], []
    raw_bf1, raw_bf2 = [], []

    for i in idx:
        ba1.append(adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]])
        bf1.append(features[:, subgraphs_1[i], :])
        raw_bf1.append(raw_features[:, subgraphs_1[i], :])

        ba2.append(adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]])
        bf2.append(features[:, subgraphs_2[i], :])
        raw_bf2.append(raw_features[:, subgraphs_2[i], :])

    # Expand adjacency: (B, K, K) -> (B, K+1, K+1)
    ba1 = torch.cat(ba1)
    ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
    ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
    ba2 = torch.cat(ba2)
    ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
    ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

    # Expand features: (B, K, F) -> (B, K+1, F)
    # Insert zero row at -2; original target moves to -1
    bf1 = torch.cat(bf1)
    bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
    bf2 = torch.cat(bf2)
    bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

    raw_bf1 = torch.cat(raw_bf1)
    raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
    raw_bf2 = torch.cat(raw_bf2)
    raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)

    return ba1, ba2, bf1, bf2, raw_bf1, raw_bf2


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the SL-GAD training loop.

    Combined loss:
        L = alpha * L_contrastive  +  beta * L_generative

        L_contrastive : BCE-with-logits over positive + cyclic-negative pairs
        L_generative  : MSE between f[:,-2,:] (context reconstruction)
                        and raw_bf[:,-1,:] (original raw target features)

    Parameters
    ----------
    model         : nn.Module
    optimiser     : torch.optim.Optimizer
    b_xent        : nn.BCEWithLogitsLoss
    mse_loss      : nn.MSELoss
    device        : torch.device
    alpha         : float  contrastive loss weight
    beta          : float  generative loss weight
    negsamp_ratio : int    negatives per positive in Discriminator
    batch_size    : int
    num_epoch     : int    max epochs
    patience      : int    early stopping patience
    subgraph_size : int    K (before expansion)
    """

    def __init__(self, model, optimiser, b_xent, mse_loss, device,
                 alpha, beta, negsamp_ratio,
                 batch_size, num_epoch, patience, subgraph_size):
        self.model = model
        self.optimiser = optimiser
        self.b_xent = b_xent
        self.mse_loss = mse_loss
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.negsamp_ratio = negsamp_ratio
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.patience = patience
        self.subgraph_size = subgraph_size

    def train(self, adj, features, raw_features, dgl_graph, nb_nodes, ft_size, expid):
        """
        Run the full training loop with early stopping.

        Parameters
        ----------
        adj, features, raw_features : torch.FloatTensor  on device
        dgl_graph     : dgl.DGLGraph
        nb_nodes      : int
        ft_size       : int
        expid         : int  used for checkpoint filename

        Returns
        -------
        best_t : int  epoch of the best saved checkpoint
        """
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/exp_{expid}.pkl'

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // self.batch_size + 1

        for epoch in range(self.num_epoch):
            self.model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            # Two independent RWR views per epoch
            subgraphs_1 = generate_rwr_subgraph(dgl_graph, self.subgraph_size)
            subgraphs_2 = generate_rwr_subgraph(dgl_graph, self.subgraph_size)

            for batch_idx in range(batch_num):
                self.optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                # 1 positive + negsamp_ratio negatives per node
                lbl = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size),
                               torch.zeros(cur_batch_size * self.negsamp_ratio))), 1
                ).to(self.device)

                ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = build_batch(
                    idx, subgraphs_1, subgraphs_2,
                    adj, features, raw_features,
                    ft_size, self.subgraph_size, self.device
                )

                logits, f_1, f_2 = self.model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)

                # Contrastive loss
                loss1 = torch.mean(self.b_xent(logits, lbl))

                # Generative loss:
                # f[:, -2, :] = decoder output at masked slot (context-only aggregation)
                # raw_bf[:, -1, :] = original raw target features
                loss2 = 0.5 * (
                    self.mse_loss(f_1[:, -2, :], raw_bf1[:, -1, :]) +
                    self.mse_loss(f_2[:, -2, :], raw_bf2[:, -1, :])
                )

                loss = self.alpha * loss1 + self.beta * loss2
                loss.backward()
                self.optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * self.batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!', flush=True)
                break

            print(f'Epoch:{epoch}  Loss:{mean_loss:.8f}', flush=True)

        return best_t
