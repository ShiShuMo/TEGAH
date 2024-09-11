#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Powered By 莫诗书
admin@52mss.cn
创建时间:2023/5/20 9:17
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from .basic_module import BasicModule, RegGRU
from torch.nn import Parameter
from .cbam import LKA_Attention_1d
from models.pygat import GraphAttentionLayer



def gen_A(num_classes, t, adj_file):
    _adj = adj_file['adj']
    _nums = adj_file['num']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, int)
    return _adj


def gen_adj(A):
    A = A.squeeze()
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GATNet(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=2048, dropout=0.6, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GATNet, self).__init__()
        self.dropout = dropout
        self.relu = nn.LeakyReLU(0.2)

        self.attentions = [GraphAttentionLayer(in_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_dim * nheads, hidden_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GAT_MODEL(BasicModule):
    def __init__(self, inp, flag, num_class, adj_file, hidden_dim, hash_dim, num_heads=8):
        super(GAT_MODEL, self).__init__()
        self.module_name = 'GAT_module'
        self.hidden_dim = hidden_dim
        self.relu = nn.LeakyReLU(0.2)
        self.inp = inp
        deta = 0.3
        if flag == "nus":
            deta = 0.01
        _adj = gen_A(num_class, deta, adj_file)
        self.A = torch.from_numpy(_adj).detach().float()
        self.adj = gen_adj(self.A).detach().cuda(non_blocking=True) 
        self.gat1 = GATNet(in_dim=num_class, hidden_dim=1024, nheads=num_heads)
        self.gat2 = GATNet(in_dim=1024, hidden_dim=hidden_dim, nheads=num_heads)
        self.conv_stage = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            LKA_Attention_1d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1)
            )

        self.gru = RegGRU(input_dim=hidden_dim, hidden_dim=hidden_dim, hidden_num_layers=2)

    def forward(self, image_feat, text_feat):
        x = torch.cat([image_feat, text_feat], dim=1)  # 200 * 4096
        x = self.gru(x)
        feat = x.view(x.size(0), self.hidden_dim, -1)
        feat = self.conv_stage(feat).squeeze()

        inp = self.gat1(self.adj, self.adj)
        inp = self.relu(inp)
        inp = self.gat2(inp, self.adj)
        x_class = torch.matmul(feat, inp.t())
        return x_class


if __name__ == '__main__':
    pass
