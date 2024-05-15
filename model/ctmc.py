import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import SingleMlp, SingleLinear
from model.loss import SupConLoss, IntraConLoss, CrossConLoss


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
    A = torch.mean(torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True))

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1  # alp 是根据真实标签 label 调整了 Dirichlet 分布参数 alpha 的新参数
    B = annealing_coef * KL(alp, c)
    return A + torch.mean(B)


def mse_loss(p, alpha, c, global_step, annealing_step=1):
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


class CTMC(nn.Module):
    def __init__(self, in_dim, num_classes, alpha, beta, lambda_epoch=50, encode_dim=[128, 256], proj_dim=[256, 128]):
        """
        :param in_dim: the dim of inputs, type is list
        :param num_classes: the class of task
        :param lambda_epoch: annealing epoch
        :param encode_dim: the dim of feature information vector, list
        :param proj_dim: the hidden dim of projection module, list
        """
        super(CTMC, self).__init__()
        self.views = len(in_dim)
        self.in_dim = in_dim
        self.encode_dim = encode_dim
        self.proj_dim = proj_dim
        self.classes = num_classes
        self.alpha = alpha
        self.lambda_epoch = lambda_epoch
        self.beta = beta
        self.FeatureInforEncoder = nn.ModuleList([SingleMlp(in_dim[i], self.encode_dim) for i in range(self.views)])
        if len(self.proj_dim) == 1:
            self.Projection = nn.ModuleList(
                [SingleLinear(self.encode_dim[-1], self.proj_dim[0]) for i in range(self.views)])
        else:
            self.Projection = nn.ModuleList([SingleMlp(self.encode_dim[-1], self.proj_dim) for i in range(self.views)])
        self.ModalityClassifier = nn.ModuleList(
            [SingleLinear(self.encode_dim[-1], self.classes) for _ in range(self.views)])
        self.softplus = nn.Softplus()

    def forward(self, X, y, global_step, mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence, project = self.infer(X)
        loss = 0
        # pseudo_out, pseudo_project = torch.flip(evidence[self.views-1], dims=[1]), torch.flip(project[self.views-1], dims=[1])
        # evidence[self.views] = pseudo_out
        # project[self.views] = pseudo_project
        if mode == 'train':
            # the loss of inter-view, for every view

            inter_criterion = SupConLoss()
            for view in range(self.views):
                inter_loss = inter_criterion(project[view], y)
                loss += self.alpha * inter_loss
            # print('inter_loss:', loss.item())

            # the loss of intra-view
            intra_criterion = IntraConLoss()
            intra_loss = intra_criterion(project)
            loss += (1 - self.alpha) * intra_loss
        alpha = dict()

        loss *= (1 - self.beta)

        # every view loss
        for view in range(self.views):
            alpha[view] = evidence[view] + 1
            loss += self.beta * ce_loss(y, alpha[view], self.classes, global_step, self.lambda_epoch)

        alpha_a = self.combine_views(alpha)
        evidence_a = alpha_a - 1  # combined evidence
        loss += self.beta * ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epoch)  # combined view loss

        return evidence, evidence_a, loss

    def infer(self, input):
        FeatureInfo, ProjectVector, Evidence = dict(), dict(), dict()
        for view in range(self.views):  # every view feature information, projection vector
            FeatureInfo[view] = self.FeatureInforEncoder[view](input[view])
            ProjectVector[view] = F.normalize(self.Projection[view](FeatureInfo[view]), p=2, dim=1)  # l2 normalize
            Evidence[view] = self.softplus(self.ModalityClassifier[view](FeatureInfo[view]))

        return Evidence, ProjectVector

    def combine_views(self, alpha):
        alpha_a = None
        for v in range(len(alpha) - 1, 0, -1):
            if alpha_a is None:
                alpha_a = combine_two(alpha[v], alpha[v - 1], self.classes)
            else:
                alpha_a = combine_two(alpha_a, alpha[v - 1], self.classes)
        return alpha_a
