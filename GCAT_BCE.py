
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

HIDDEN = 1024
LAYER = 3  # 1-5
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, lamda, alpha, layer_idx):
        theta = min(1, math.log(lamda / layer_idx + 1))
        hi = torch.spmm(adj, x)
        support = (1 - alpha) * hi + alpha * h0
        output = theta * torch.mm(support, self.weight) + (1 - theta) * support
        return output


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj, h0, lamda, alpha, layer_idx):
        h = torch.mm(x, self.W)
        N = h.size(0)

        a_input = torch.cat([
            h.repeat(1, N).view(N * N, -1),
            h.repeat(N, 1)
        ], dim=1).view(N, N, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        theta = min(1, math.log(lamda / layer_idx + 1))
        hi = torch.mm(attention, x)
        support = (1 - alpha) * hi + alpha * h0
        output = theta * torch.mm(support, self.W) + (1 - theta) * support
        return output


class GCN_GAT(nn.Module):
    def __init__(self, nlayers=LAYER, nfeat=90, nhidden=HIDDEN, dropout=DROPOUT,
                 lamda=LAMBDA, alpha=ALPHA, out_dim=1):
        super(GCN_GAT, self).__init__()
        self.fc_in = nn.Linear(nfeat, nhidden)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(nhidden, nhidden) for _ in range(nlayers)
        ])
        self.gat = GATLayer(nhidden, nhidden, dropout, 0.2)

        self.fc_hidden = nn.Linear(nhidden, 64)
        self.fc_out = nn.Linear(64, out_dim)

        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.act_fn = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, dist_adj, full_adj):
        h0 = self.act_fn(self.fc_in(F.dropout(x, self.dropout, training=self.training)))

        h = h0
        for i, conv in enumerate(self.gcn_layers):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act_fn(conv(h, dist_adj, h0, self.lamda, self.alpha, i + 1))

        h = F.dropout(h, self.dropout, training=self.training)
        h = F.elu(self.gat(h, full_adj, h0, self.lamda, self.alpha, len(self.gcn_layers) + 1))

        h = self.act_fn(self.fc_hidden(h))
        # out = self.sigmoid(self.fc_out(h))
        out = self.fc_out(h)
        return out.view(-1)  # safer than squeeze()

