# -*- coding:utf-8 -*-
# @Time    : 2024/1/1 11:50
# @Author  : Shukang Zhang
# @FileName: ecamvf.py
# @Software: PyCharm
# @Description: this is a program related to
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import SingleMlp, SingleLinear
from model.loss import SupConLoss, IntraConLoss, CrossConLoss
from model.camvf import CAMVF


class ECAMVF(CAMVF):
    def __init__(self, *args):
        """
        :param in_dim: the dim of inputs, type is list
        :param num_classes: the class of task
        :param lambda_epoch: annealing epoch
        :param encode_dim: the dim of feature information vector, list
        :param proj_dim: the hidden dim of projection module, list
        """
        super(ECAMVF, self).__init__(*args)
        last_size = self.classes * self.views
        self.clf_FeatureInforEncoder = SingleMlp(last_size, self.encode_dim)
        if len(self.proj_dim) == 1:
            self.clf_Projection = SingleLinear(self.encode_dim[-1], self.proj_dim[0])
        else:
            self.clf_Projection = SingleMlp(self.encode_dim[-1], self.proj_dim)
        self.clf_ModalityClassifier = SingleLinear(self.encode_dim[-1], self.classes)

    def forward(self, X, y, global_step, mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence, project = self.infer(X)
        loss = 0
        evidence_list = []
        for view in range(self.views):
            evidence_list.append(evidence[view])
        pseudo_out = torch.cat(evidence_list, dim=1)
        pseudo_out, pseudo_project = self.infer_eqmuf(pseudo_out)
        evidence[self.views] = pseudo_out
        project[self.views] = pseudo_project
        # if mode == 'train':
        #     # the loss of inter-view, for every view
        #     inter_criterion = SupConLoss()
        #     for view in range(self.views + 1):
        #         inter_loss = inter_criterion(project[view], y)
        #         loss += self.alpha * inter_loss
        #     # print('inter_loss:', loss.item())
        #
        #     # the loss of intra-view
        #     intra_criterion = IntraConLoss()
        #     intra_loss = intra_criterion(project)
        #     loss += (1 - self.alpha) * intra_loss
            # print('intra_loss:', intra_loss.item())
            # the loss of cross-view
            # cross_criterion = CrossConLoss()
            # loss += cross_criterion(evidence[view], y)
            # the loss of cross-view

        # every view loss
        loss *= (1 - self.beta)
        alpha = dict()
        evidence_u = dict()
        p = dict()
        step_rate = (global_step + 1) / self.lambda_epoch

        for view in range(self.views + 1):
            alpha[view] = evidence[view] + 1
            loss += self.beta * self.ce_loss(y, alpha[view], self.classes, global_step, self.lambda_epoch, c_KL=True)
            # loss += self.beta * nn.CrossEntropyLoss()(evidence[view], y)

            # evidential
            S = torch.sum(alpha[view], dim=1, keepdim=True)
            E = alpha[view] - 1
            b = E / (S.expand(E.shape))
            u = self.classes / S  # 不确定性
            p[view] = alpha[view] / S.expand(alpha[view].shape)  # 概率
            p[view] = torch.cat([b], dim=1)
            evidence_u[view] = u

        evidence_a, evidence_u_w = self.combine_views(p, evidence_u, evidence)
        # mask = evidence_u_w[:, None] > 0.5
        alpha_a = evidence_a + 1
        # alpha_a = mask * alpha_a + (~mask) * self.combine_views(alpha)

        loss += self.beta * self.ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epoch,
                                                             c_KL=True)  # combined view loss
        # loss += self.beta * nn.CrossEntropyLoss()(evidence_a,y)

        return evidence, evidence_a, loss

    def infer_eqmuf(self, input):
        ProjectVector=  None
        FeatureInfo = self.clf_FeatureInforEncoder(input)
        # ProjectVector = F.normalize(self.clf_Projection(FeatureInfo), p=2, dim=1)  # l2 normalize
        Evidence = self.softplus(self.clf_ModalityClassifier(FeatureInfo))

        return Evidence, ProjectVector
