# -*- coding:utf-8 -*-
# @Time    : 2024/1/1 12:13
# @Author  : Yinkai Yang
# @FileName: mlp.py
# @Software: PyCharm
# @Description: this is a program related to
import torch.nn as nn


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(1.0) if m.bias is not None else None


class SingleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        :param in_dim: dim of input
        :param out_dim: dim of output
        """
        super(SingleLinear, self).__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim),nn.BatchNorm1d(out_dim),nn.ReLU())
        self.clf.apply(xavier_init)

    def forward(self, x):
        return self.clf(x)


class SingleMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        """
        :param in_dim: a number
        :param hidden_dim: a list, length at least 1
        """
        super(SingleMlp, self).__init__()
        self.dim = [in_dim] + hidden_dim
        self.length = len(self.dim)
        layers = []
        for i in range(self.length - 1):
            layers.append(nn.Linear(self.dim[i], self.dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.dim[i + 1]))
            layers.append(nn.ReLU())
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
