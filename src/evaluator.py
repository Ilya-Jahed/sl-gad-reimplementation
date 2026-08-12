"""
src/evaluator.py
----------------
Multi-round anomaly scoring and AUC evaluation for SL-GAD.

Handles:
  - Multi-round stochastic inference over random subgraph views
  - Contrastive anomaly score from Discriminator logits
  - Generative anomaly score from attribute reconstruction distance
  - MinMax-normalised score fusion controlled by alpha / beta
  - Final ROC-AUC computation

Forked and modularised from GRAND-Lab/CoLA run.py
"""

import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from src.dataset import generate_rwr_subgraph
from src.trainer import build_batch


class Evaluator:
    """
    Runs multi-round stochastic inference and computes the final AUC.

    Anomaly scores:
        ano_score_1  (contrastive) = -(pos_logit - neg_logit)
                                     Low agreement with context -> anomalous
        ano_score_2  (generative)  = L2 distance (reconstructed vs original)
                                     High reconstruction error -> anomalous

    Fusion:
        ano_score = alpha * normalise(ano_score_1) + beta * normalise(ano_score_2)

    Final per-node score is the mean across all inference rounds.

    Parameters
    ----------
    model           : nn.Module  trained SL-GAD Model
    device          : torch.device
    alpha           : float  contrastive score weight
    beta            : float  generative score weight
    negsamp_ratio   : int    must match training value
    batch_size      : int
    subgraph_size   : int    K (before expansion)
    auc_test_rounds : int    number of stochastic inference rounds
    """

    def __init__(self, model, device,
                 alpha, beta, negsamp_ratio,
                 batch_size, subgraph_size, auc_test_rounds):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.negsamp_ratio = negsamp_ratio
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.auc_test_rounds = auc_test_rounds

    def _compute_ano_score(self, logits, dist, cur_batch_size):
        """
        Compute fused anomaly score for a single batch.

        Parameters
        ----------
        logits         : torch.Tensor  sigmoid logits, shape (B*(1+negsamp),)
        dist           : torch.Tensor  L2 distance,    shape (B,)
        cur_batch_size : int

        Returns
        -------
        ano_score : np.ndarray  shape (B,)
        """
        if self.alpha != 0.0 and self.beta != 0.0:
            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()

            if self.negsamp_ratio == 1:
                ano_score_1 = -(logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            else:
                pos_score = logits[:cur_batch_size]
                neg_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                ano_score_1 = -(pos_score - neg_score).cpu().numpy()

            ano_score_2 = dist.cpu().numpy()
            ano_score_1 = scaler1.fit_transform(ano_score_1.reshape(-1, 1)).reshape(-1)
            ano_score_2 = scaler2.fit_transform(ano_score_2.reshape(-1, 1)).reshape(-1)
            ano_score = self.alpha * ano_score_1 + self.beta * ano_score_2

        elif self.alpha != 0.0 and self.beta == 0.0:
            if self.negsamp_ratio == 1:
                ano_score = -(logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            else:
                pos_score = logits[:cur_batch_size]
                neg_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                ano_score = -(pos_score - neg_score).cpu().numpy()

        elif self.alpha == 0.0 and self.beta != 0.0:
            ano_score = dist.cpu().numpy()

        else:
            raise ValueError("alpha and beta cannot both be zero.")

        return ano_score

    def evaluate(self, adj, features, raw_features,
                 dgl_graph, nb_nodes, ft_size, ano_label):
        """
        Run multi-round stochastic inference and return the final ROC-AUC.

        Parameters
        ----------
        adj, features, raw_features : torch.FloatTensor  on device
        dgl_graph     : dgl.DGLGraph
        nb_nodes      : int
        ft_size       : int
        ano_label     : np.ndarray  binary ground-truth (1 = anomaly)

        Returns
        -------
        auc : float  ROC-AUC score
        """
        batch_num = nb_nodes // self.batch_size + 1
        multi_round_ano_score = np.zeros((self.auc_test_rounds, nb_nodes))

        print('Testing AUC!', flush=True)

        for round_idx in range(self.auc_test_rounds):
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            # Fresh random views per round to reduce score variance
            subgraphs_1 = generate_rwr_subgraph(dgl_graph, self.subgraph_size)
            subgraphs_2 = generate_rwr_subgraph(dgl_graph, self.subgraph_size)

            for batch_idx in range(batch_num):
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = build_batch(
                    idx, subgraphs_1, subgraphs_2,
                    adj, features, raw_features,
                    ft_size, self.subgraph_size, self.device
                )

                with torch.no_grad():
                    logits, dist = self.model.inference(
                        bf1, bf2, raw_bf1, raw_bf2, ba1, ba2
                    )
                    logits = torch.sigmoid(torch.squeeze(logits))

                ano_score = self._compute_ano_score(logits, dist, cur_batch_size)
                multi_round_ano_score[round_idx, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0)
        auc = roc_auc_score(ano_label, ano_score_final)
        return auc
