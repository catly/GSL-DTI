# -*- coding: utf-8 -*-
from utilsdtiseed import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from GCNLayer import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu).apply(init))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout)
        )
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()
        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout))
    def forward(self, s_g, s_h_1, s_h_2):
        h1 = self.sum_layers[0](s_g[0], s_h_1)
        h2 = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2

class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 128)
        self.dropout = dropout
    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res

class ENCODER(nn.Module):
    def __init__(self, nfeat,dim):
        super(ENCODER, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, dim, bias=False))
    def forward(self, x):
        output = self.MLP(x)
        return output

class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),
            nn.LogSoftmax(dim=1))
    def forward(self, x):
        output = self.MLP(x)
        return output


class GSLDTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout,dim):
        super(GSLDTI, self).__init__()
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout)
        self.GCN = GCN(256, dropout)
        self.ENCODER = ENCODER(256,dim)
        self.MLP = MLP(128)

    def forward(self, graph, h, dateset_index, data,iftrain=True, d=None, p=None):
        if iftrain:
            d, p= self.HAN_DTI(graph, h[0], h[1])
        feature = torch.cat((d[data[:, :1]], p[data[:, 1:2]]), dim=2).squeeze(1)
        X = self.ENCODER(feature)
        t=X.mean()
        am = ((X @ X.T) >t).type(torch.int)
        row, col = np.diag_indices_from(am)
        am[row, col] = 1
        am = am.cpu().numpy()

        edge = np.nonzero(am)
        edge = load_graph(np.array(edge), data.shape[0])
        feature1 = self.GCN(feature, edge)

        pred = self.MLP(feature1)
        pred1 = pred[dateset_index]


        if iftrain:
            return pred1,  d, p
        else:
            return pred1


def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
