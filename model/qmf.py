# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: QMUF
# @Author      : Shukang Zhang
# @Owner       : fusheng
# @Data        : 2024/1/19
# @Time        : 12:06
# @Description : qmf module
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import SingleMlp, SingleLinear
from model.loss import SupConLoss, IntraConLoss, CrossConLoss


def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1, 1))

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                                    rank_input2,
                                                    -rank_target.reshape(-1, 1))
    if torch.isnan(ranking_loss):
        print('nan')

    return ranking_loss


class QMF(nn.Module):
    def __init__(self, in_dim, num_classes, alpha, lambda_epoch=50, encode_dim=[128, 256], proj_dim=[256, 128]):
        """
        :param in_dim: the dim of inputs, type is list
        :param num_classes: the class of task
        :param lambda_epoch: annealing epoch
        :param encode_dim: the dim of feature information vector, list
        :param proj_dim: the hidden dim of projection module, list
        """
        super(QMF, self).__init__()
        self.views = len(in_dim)
        self.in_dim = in_dim
        self.encode_dim = encode_dim
        self.proj_dim = proj_dim
        self.classes = num_classes
        self.alpha = alpha
        self.lambda_epoch = lambda_epoch

        self.FeatureInforEncoder = nn.ModuleList([SingleMlp(in_dim[i], self.encode_dim) for i in range(self.views)])
        if len(self.proj_dim) == 1:
            self.Projection = nn.ModuleList(
                [SingleLinear(self.encode_dim[-1], self.proj_dim[0]) for i in range(self.views)])
        else:
            self.Projection = nn.ModuleList([SingleMlp(self.encode_dim[-1], self.proj_dim) for i in range(self.views)])
        self.ModalityClassifier = nn.ModuleList(
            [SingleLinear(self.encode_dim[-1], self.classes) for _ in range(self.views)])
        self.softplus = nn.Softplus()

    def forward(self, X, y,mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence = self.infer(X)
        loss = 0

        energy = dict()
        conf = dict()
        evidence_merge = None
        for view in range(self.views):
            loss += nn.CrossEntropyLoss()(evidence[view], y)

            energy[view] = torch.log(torch.sum(torch.exp(evidence[view])+1e-3, dim=1))
            energy[view] = torch.where(torch.isinf(energy[view]), torch.full_like(energy[view], 0), energy[view])

            conf[view] = energy[view] / 10
            conf[view] = torch.reshape(conf[view], (-1, 1))
            if evidence_merge is None:
                evidence_merge = evidence[view] * conf[view].detach()
            else:
                evidence_merge += evidence[view] * conf[view].detach()

        clf_loss = loss + nn.CrossEntropyLoss()(evidence_merge, y)


        return evidence, evidence_merge, clf_loss,conf

    def infer(self, input):
        FeatureInfo, Evidence = dict(), dict()
        for view in range(self.views):  # every view feature information, projection vector
            FeatureInfo[view] = self.FeatureInforEncoder[view](input[view])
            Evidence[view] = self.softplus(self.ModalityClassifier[view](FeatureInfo[view]))

        return Evidence
