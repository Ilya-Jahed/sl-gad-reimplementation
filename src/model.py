import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)


class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    Weighted Sum Readout using attention scores between context nodes and a query vector.
    """
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    Bilinear discriminator with in-batch negative sampling via cyclic shift of context summaries.
    """
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn_enc = GCN(n_in, n_h, activation)
        self.gcn_dec = GCN(n_h, n_in, activation)

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc1 = Discriminator(n_h, negsamp_round)
        self.disc2 = Discriminator(n_h, negsamp_round)
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, seq1, seq2, seq3, seq4, adj1, adj2, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
        """
        Training Forward Pass.
        
        Input Mappings (from build_batch):
        - seq1, seq2 : Masked features (bf1, bf2) -> Used for Contrastive Path
        - seq3, seq4 : Raw/Unmasked features (raw_bf1, raw_bf2) -> Used for Generative Path
        - adj1, adj2 : Expanded adjacency matrices (ba1, ba2)
        """
        # ==========================================
        # 1. ENCODING (Shared weights for all views)
        # ==========================================
        # Contrastive Path: Encode masked features
        h_1 = self.gcn_enc(seq1, adj1, sparse)
        h_2 = self.gcn_enc(seq2, adj2, sparse)
        
        # Generative Path: Encode raw features
        h_3 = self.gcn_enc(seq3, adj1, sparse)
        h_4 = self.gcn_enc(seq4, adj2, sparse)

        # ==========================================
        # 2. GENERATIVE PATH (Decoding)
        # ==========================================
        # Decode raw representations to reconstruct features
        # f_1, f_2 will be compared with raw features outside (in trainer) via MSELoss
        f_1 = self.gcn_dec(h_3, adj1, sparse)
        f_2 = self.gcn_dec(h_4, adj2, sparse)

        # ==========================================
        # 3. CONTRASTIVE PATH (Target Isolation & Readout)
        # ==========================================
        if self.read_mode != 'weighted_sum':
            # Extract pure target node representations (isolated at index -1)
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            
            # Summarize context nodes (indices 0 to -2) into a single vector 'c'
            c1 = self.read(h_1[:, :-1, :])
            c2 = self.read(h_2[:, :-1, :])
        else:
            # Weighted sum requires the ghost/mask node (index -2) as the query vector
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            c2 = self.read(h_2[:, :-1, :], h_2[:, -2:-1, :])

        # ==========================================
        # 4. DISCRIMINATION (Cross-View Prediction)
        # ==========================================
        # Compare context of View 1 with Target of View 2 (and vice versa)
        ret1 = self.disc1(c1, h_mv_2, samp_bias1, samp_bias2)
        ret2 = self.disc2(c2, h_mv_1, samp_bias1, samp_bias2)
        
        # Average the predictions from both discriminators
        ret = torch.cat((ret1, ret2), dim=-1).mean(dim=-1).unsqueeze(dim=-1)
        
        # Returns: Contrastive Logits (ret), Reconstructed Features (f_1, f_2)
        return ret, f_1, f_2


    def inference(self, seq1, seq2, seq3, seq4, adj1, adj2, sparse=False):
        """
        Evaluation Inference Pass.
        Similar to forward(), but computes the generative distance (dist) internally.
        """
        # 1. ENCODING
        h_1 = self.gcn_enc(seq1, adj1, sparse)
        h_2 = self.gcn_enc(seq2, adj2, sparse)
        h_3 = self.gcn_enc(seq3, adj1, sparse)
        h_4 = self.gcn_enc(seq4, adj2, sparse)

        # 2. DECODING
        f_1 = self.gcn_dec(h_3, adj1, sparse)
        f_2 = self.gcn_dec(h_4, adj2, sparse)

        # ==========================================
        # 3. GENERATIVE ANOMALY SCORE COMPUTATION
        # ==========================================
        # Calculate L2 distance between the decoder's guess at the masked slot (index -2)
        # and the true raw features at the isolated slot (index -1)
        dist1 = self.pdist(f_1[:, -2, :], seq3[:, -1, :])
        dist2 = self.pdist(f_2[:, -2, :], seq4[:, -1, :])
        dist = 0.5 * (dist1 + dist2) # Average reconstruction error across views

        # 4. CONTRASTIVE PATH (Same as forward)
        if self.read_mode != 'weighted_sum':
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :])
            c2 = self.read(h_2[:, :-1, :])
        else:
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            c2 = self.read(h_2[:, :-1, :], h_2[:, -2:-1, :])

        # 5. DISCRIMINATION (Same as forward, but no bias sampling needed for test)
        ret1 = self.disc1(c1, h_mv_2, None, None)
        ret2 = self.disc2(c2, h_mv_1, None, None)
        ret = torch.cat((ret1, ret2), dim=-1).mean(dim=-1).unsqueeze(dim=-1)
        
        # Returns: Contrastive Logits (ret), Generative Error/Distance (dist)
        return ret, dist


# Alias kept for any future references
SLGADModel = Model