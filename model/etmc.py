import torch
from torch import nn

from model.ctmc import CTMC, ce_loss
from model.loss import SupConLoss, IntraConLoss
from model.mlp import SingleMlp, SingleLinear
import torch.nn.functional as F


class ETMC(CTMC):
    def __init__(self, *args):
        super(ETMC, self).__init__(*args)
        last_size = self.classes * self.views
        self.clf = nn.ModuleList([SingleMlp(self.in_dim[i], self.encode_dim) for i in range(self.views)])
        self.clf_FeatureInforEncoder = SingleMlp(last_size, self.encode_dim)
        if len(self.proj_dim) == 1:
            self.clf_Projection = SingleLinear(self.encode_dim[-1], self.proj_dim[0])
        else:
            self.clf_Projection = SingleMlp(self.encode_dim[-1], self.proj_dim)
        self.clf_ModalityClassifier = SingleLinear(self.encode_dim[-1], self.classes)

    def forward(self, X, y, global_step, mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence, project = self.infer(X)
        evidence_list = []
        for view in range(self.views):
            evidence_list.append(evidence[view])
        pseudo_out = torch.cat(evidence_list, dim=1)
        pseudo_out, pseudo_project = self.infer_etmc(pseudo_out)
        evidence[self.views] = pseudo_out
        project[self.views] = pseudo_project
        loss = 0
        alpha = dict()
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
            # loss += cross_criterion(project, y)
        # every view loss
        loss *= (1 - self.beta)

        for view in range(self.views + 1):
            alpha[view] = evidence[view] + 1
            loss += self.beta * ce_loss(y, alpha[view], self.classes, global_step, self.lambda_epoch)

        alpha_a = self.combine_views(alpha)
        evidence_a = alpha_a - 1  # combined evidence
        loss += self.beta * ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epoch)  # combined view loss

        return evidence, evidence_a, loss

    def infer_etmc(self, input):
        ProjectVector = None
        FeatureInfo = self.clf_FeatureInforEncoder(input)
        # ProjectVector = F.normalize(self.clf_Projection(FeatureInfo), p=2, dim=1)  # l2 normalize
        Evidence = self.softplus(self.clf_ModalityClassifier(FeatureInfo))

        return Evidence, ProjectVector
