# -*- coding:utf-8 -*-
# @Time    : 2024/1/1 11:50
# @Author  : Shukang Zhang
# @FileName: ctmc.py
# @Software: PyCharm
# @Description: this is a program related to
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import SingleMlp, SingleLinear
from model.loss import SupConLoss, IntraConLoss, CrossConLoss
from scipy.spatial.distance import pdist


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


def combine_two(alpha1, alpha2, classes):
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))
        u[v] = classes / S[v]  # 不确定性

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


class CAMVF(nn.Module):
    def __init__(self, in_dim, num_classes, alpha, beta, lambda_epoch=50, encode_dim=None, proj_dim=None,
                 exp_eta=0.5):
        """
        :param in_dim: the dim of inputs, type is list
        :param num_classes: the class of task
        :param lambda_epoch: annealing epoch
        :param encode_dim: the dim of feature information vector, list
        :param proj_dim: the hidden dim of projection module, list
        """
        super(CAMVF, self).__init__()
        if proj_dim is None:
            proj_dim = [256, 128]
        if encode_dim is None:
            encode_dim = [256, 512]
        self.views = len(in_dim)
        self.encode_dim = encode_dim
        self.proj_dim = proj_dim
        self.classes = num_classes
        self.alpha = alpha
        self.lambda_epoch = lambda_epoch
        self.exp_eta = exp_eta
        self.beta = beta
        self.in_dim = in_dim

        self.FeatureInforEncoder = nn.ModuleList([SingleMlp(in_dim[i], self.encode_dim) for i in range(self.views)])
        if len(self.proj_dim) == 1:
            self.Projection = nn.ModuleList(
                [SingleLinear(self.encode_dim[-1], self.proj_dim[0]) for i in range(self.views)])
        else:
            self.Projection = nn.ModuleList([SingleMlp(self.encode_dim[-1], self.proj_dim) for i in range(self.views)])
        self.ModalityClassifier = nn.ModuleList(
            [SingleLinear(self.encode_dim[-1], self.classes) for _ in range(self.views)])
        self.softplus = nn.Softplus()

    def ce_loss(self, p, alpha, c, global_step, annealing_step, c_KL=True):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        A = torch.mean(torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True))
        #
        if c_KL:
            annealing_coef = min(1, global_step / annealing_step)
            alp = E * (1 - label) + 1  # alp 是根据真实标签 label 调整了 Dirichlet 分布参数 alpha 的新参数
            B = annealing_coef * KL(alp, c)
            return A + torch.mean(B)
        return A

    def mse_loss(self, p, alpha, c, global_step, annealing_step=1, c_KL=True):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S
        label = F.one_hot(p, num_classes=c)
        A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        annealing_coef = min(1, global_step / annealing_step)
        alp = E * (1 - label) + 1
        C = annealing_coef * KL(alp, c)
        return torch.mean((A + B) + C)

    def jousselme_distance(self, m1, m2):
        dis = m1 - m2
        dis = dis.unsqueeze(1)
        return torch.sqrt((torch.bmm(dis, torch.transpose(dis, -1, 1))) / 2)

    def count_DM(self, views):
        dist_matrix = torch.zeros((len(views), len(views), views[0].shape[0]), device=views[0].device)
        for i in range(len(views)):
            for j in range(len(views)):
                # dist_matrix[i][j] = nn.PairwiseDistance(p=2)(views[i], views[j]).squeeze()
                dist_matrix[i][j] = self.jousselme_distance(views[i], views[j]).squeeze()
        return dist_matrix.permute(2, 0, 1)

    def count_normalize_Deng_entropy(self, views):
        deng_entropy_matrix = torch.zeros((len(views), views[0].shape[0]), device=views[0].device)
        max_entropy = 0
        for i in range(len(views)):
            x_modified = torch.where(views[i] == 0, torch.ones_like(views[i]), views[i])
            deng_entropy_matrix[i] = -torch.sum(views[i] * torch.log2(x_modified), dim=1).squeeze()
        deng_entropy_matrix = deng_entropy_matrix.permute(1, 0)
        return torch.div(deng_entropy_matrix, torch.sum(deng_entropy_matrix, dim=1)[:, None])

    def forward(self, X, y, global_step, mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence, project = self.infer(X)
        loss = 0

        alpha = dict()
        step_rate = global_step / self.lambda_epoch

        # every view loss
        alpha = dict()
        evidence_u = dict()
        p = dict()

        # for view in range(self.views):
        #     alpha[view] = evidence[view] + 1
        #     loss += self.beta * self.ce_loss(y, alpha[view], self.classes, global_step,
        #                                                              self.lambda_epoch,
        #                                                              c_KL=True)
        #     # loss += self.beta * nn.CrossEntropyLoss()(evidence[view], y)
        #
        #     S = torch.sum(alpha[view], dim=1, keepdim=True)
        #     E = alpha[view] - 1
        #     b = E / (S.expand(E.shape))
        #     u = self.classes / S
        #     p[view] = alpha[view] / S.expand(alpha[view].shape)
        #     p[view] = torch.cat([p[view]], dim=1)
        #     evidence_u[view] = u
        #
        # evidence_a, evidence_u_w = self.combine_views(p, evidence_u, evidence)
        # alpha_a = evidence_a + 1
        #
        # loss += min(step_rate,1)* self.beta * self.ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epoch,
        #                                                      c_KL=True)
        evidence_a = torch.zeros_like(evidence[0])
        for view in range(self.views):
            max_idx = torch.argmax(evidence[view], dim=1)
            loss += nn.CrossEntropyLoss()(evidence[view], y)
            one_hot_A = torch.zeros_like(evidence[view])
            one_hot_A.scatter_(1, max_idx.unsqueeze(1), 1)

            evidence_a += one_hot_A
        evidence_a = evidence_a / self.views
        evidence_a.requires_grad = True
        loss += nn.CrossEntropyLoss()(evidence_a, y)
        return evidence, evidence_a, loss

    def infer(self, input):
        FeatureInfo, ProjectVector, Evidence = dict(), dict(), dict()
        for view in range(self.views):  # every view feature information, projection vector
            FeatureInfo[view] = self.FeatureInforEncoder[view](input[view])
            # ProjectVector[view] = F.normalize(self.Projection[view](FeatureInfo[view]), p=2, dim=1)  # l2 normalize
            Evidence[view] = self.softplus(self.ModalityClassifier[view](FeatureInfo[view]))

        return Evidence, ProjectVector

    def combine_views(self, alpha, evidence_u, evidence):
        # Step1 计算距离矩阵
        dist_matrix = self.count_DM(alpha)
        # Step2 计算各证据的平均距离
        dist_matrix_avg = torch.sum(dist_matrix, dim=1) / (len(alpha) - 1)

        # Step3 计算全局证据距离
        dist_matrix_global_avg = torch.mean(dist_matrix_avg, dim=1)
        # Step4 Step5  计算归一化邓熵
        # deng_entropy = self.count_normalize_Deng_entropy(alpha)
        # deng_entropy_max, _ = torch.max(deng_entropy, dim=1)
        # 归一化不确定性估计
        evidence_u_matrix = torch.zeros((evidence_u[0].shape[0], len(evidence_u)), device=evidence_u[0].device)  # batch
        evidence_u_sum = torch.zeros((evidence_u[0].shape[0], 1), device=evidence_u[0].device)
        for view in range(len(evidence_u)):
            evidence_u_sum += evidence_u[view]
        for view in range(len(evidence_u)):
            evidence_u_matrix[:, view][:, None] = evidence_u[view] / evidence_u_sum
        wight = evidence_u_matrix
        wight_max, _ = torch.max(wight, dim=1)

        # Step6 计算归一化权重
        wight_normalize = torch.zeros((alpha[0].shape[0], len(alpha)), device=alpha[0].device)
        mask = dist_matrix_avg < dist_matrix_global_avg[:, None]

        wight_normalize = torch.exp(-((wight_max + 1)[:, None] - wight)) * (~mask) + torch.exp(
            -wight) * mask
        # wight_normalize = torch.exp(-wight)
        wight_normalize = wight_normalize / torch.sum(wight_normalize, dim=1)[:, None]
        # Step7 综合证据
        wight_normalize_exp = torch.exp(wight_normalize / self.exp_eta)
        # wight_normalize_exp = wight_normalize
        wight_normalize_exp_sum = torch.sum(wight_normalize_exp, dim=1)[:, None]
        # wight_normalize_exp = wight_normalize_exp / torch.sum(wight_normalize_exp, dim=1)[:, None]
        evidence_w = torch.zeros((len(alpha), alpha[0].shape[0], evidence[0].shape[1]),
                                 device=alpha[0].device)  # view * batch * class
        evidence_u_w = torch.zeros((len(alpha), evidence_u[0].shape[0]),
                                   device=alpha[0].device)  # view * batch * class
        wight_normalize_exp = torch.ones((alpha[0].shape[0], len(alpha)), device=alpha[0].device)

        for view in range(len(evidence)):
            evidence_w[view] = (evidence[view] * wight_normalize_exp[:, view][:, None])
            evidence_u_w[view][:, None] = ((evidence_u[view]
                                            * wight_normalize_exp[:, view][:, None])
            )
        evidence_w_sum = torch.sum(evidence_w, dim=0)
        e = evidence_w_sum / torch.sum(wight_normalize_exp, dim=1)[:, None]
        # e = self.combine_views_ctmc(evidence_w+1)-1
        return e, torch.sum(evidence_u_w, dim=0)[:, None] / torch.sum(wight_normalize_exp, dim=1)[:, None]

    def combine_views_ctmc(self, alpha):
        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = combine_two(alpha[0], alpha[1], self.classes)
            else:
                alpha_a = combine_two(alpha_a, alpha[v + 1], self.classes)
        return alpha_a
