# -*- coding:utf-8 -*-
# @Time    : 2023/12/8 22:19
# @Author  : Yinkai Yang
# @FileName: atmc.py
# @Software: PyCharm
# @Description: this is a program related to
import torch
import torch.nn as nn
import torch.nn.functional as F
from ATMC.model.linear import SingleLinear
from ATMC.model.attention import Attention


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1  # alp 是根据真实标签 label 调整了 Dirichlet 分布参数 alpha 的新参数
    B = annealing_coef * KL(alp, c)
    return (A + B)
    # return A


def combine_two(alpha1, alpha2, classes):
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))
        u[v] = classes / S[v]

    # b^0 @ b^(0+1)
    bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
    # b^0 * u^1
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    # b^1 * u^0
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    # calculate C
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    C = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
    # calculate u^a
    u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

    # calculate new S
    S_a = classes / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    return alpha_a


class ATMC(nn.Module):
    def __init__(self, in_dim, num_classes, dropout, lambda_epoch=1, hidden_dim=[1000]):
        super(ATMC, self).__init__()
        self.views = len(in_dim)
        self.hidden_dim = hidden_dim
        self.classes = num_classes
        self.dropout_rate = dropout
        self.lambda_epoch = lambda_epoch

        self.FeatureInforEncoder = nn.ModuleList([SingleLinear(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.AttentionBlock = nn.ModuleList(
            [Attention(in_dim[0], in_dim[view], hidden_dim=in_dim[view]) for view in range(1, self.views)]
        )
        self.FeatureEncoder = nn.ModuleList([SingleLinear(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.ModalityClassifier = nn.ModuleList([SingleLinear(hidden_dim[0], self.classes) for _ in range(self.views)])
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.softplus = nn.Softplus()

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        # every view loss
        for view in range(self.views):
            alpha[view] = evidence[view] + 1
            loss += ce_loss(y, alpha[view], self.classes, global_step, self.lambda_epoch)

        alpha_a = self.combine_views(alpha)
        evidence_a = alpha_a - 1  # combined evidence
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epoch)  # combined view loss
        loss = torch.mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        FeatureInfo, feature, Guided_view, Prediction = dict(), dict(), dict(), dict()

        # the first part: feature information
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](input[view]))  # no using dropout
            feature[view] = input[view] * FeatureInfo[view]

        # the second part: guided view
        for view in range(self.views):
            if view == 0:
                Guided_view[view] = input[view]
            else:
                Guided_view[view], _ = self.AttentionBlock[view - 1](input[0], input[view])
                Guided_view[view] = torch.sigmoid(Guided_view[view])

        # feature encoder
        for view in range(self.views):
            feature[view] = torch.mean(torch.stack([feature[view], Guided_view[view]]), dim=0)  # add two parts
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = self.dropout(feature[view])

        # Classifier
        for view in range(self.views):
            Prediction[view] = self.softplus(self.ModalityClassifier[view](feature[view]).squeeze(0))

        return Prediction

    def combine_views(self, alpha):
        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = combine_two(alpha[0], alpha[1], self.classes)
            else:
                alpha_a = combine_two(alpha_a, alpha[v + 1], self.classes)
        return alpha_a
