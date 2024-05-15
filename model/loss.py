# -*- coding:utf-8 -*-
# @Time    : 2024/1/1 20:09
# @Author  : Yinkai Yang
# @FileName: loss.py
# @Software: PyCharm
# @Description: this is a program related to

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    # name shuould be InterConLoss
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        :param features: hidden vector of shape [bs, dim].
        :param labels: ground truth of shape [bs].
        :return:  a loss scalar.
        """
        device = torch.device('cuda')

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)  # bs * 1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # bs * bs

        contrast_feature = features  # # bs * out_dims
        anchor_feature = contrast_feature  # bs * out_dims

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )  # diagonal of bs * bs matrix is 0
        mask = mask * logits_mask  # not consider diagonal elements, set 0

        # compute log_prob
        eps = 1e-6
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)  # logits - log(sum of row)
        # print('log_prob', log_prob)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # avoid division by 0
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print('-------------------------------------------loss\n', loss)
        loss = loss.mean()  # mean over anchor_count, shape is 1

        return loss


class IntraConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(IntraConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features):
        """
        :param features: a dict contains three views, view's feature shape is [bs, dim].
        :return: the loss of intra-views.
        """
        device = torch.device('cuda')
        batch_size = features[0].shape[0]

        diag_mask = torch.eye(batch_size).to(device)
        # undiag_mask = torch.ones(batch_size, batch_size).to(device) - diag_mask

        loss = 0

        eps = 1e-6
        for view in range(len(features)):
            anchor_feature = features[view]
            neg_exp_sim_all = 0
            pos_sim_all = 0
            for con in range(1, len(features)):
                view_tmp = (view + con) % len(features)
                contrast_feature = features[view_tmp]
                anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
                # for numerical stability
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()

                pos_sim_all += (diag_mask * anchor_dot_contrast).sum(1)

                exp_sim = torch.exp(anchor_dot_contrast)

                neg_exp_sim_all += exp_sim.sum(1)

            l_tmp = pos_sim_all - 2 * torch.log(neg_exp_sim_all + eps)
            loss_tmp = l_tmp.sum(0)
            loss += loss_tmp
        loss = - (self.temperature / self.base_temperature) * loss / batch_size
        return loss


class CrossConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CrossConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        :param features: a dict contains three views, view's feature shape is [bs, dim].
        :param labels: the label of samples, shape is [bs].
        :return: the loss of cross-views.
        """
        device = torch.device('cuda')
        batch_size = features[0].shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)  # bs * 1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # bs * bs

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  # not consider diagonal elements, set 0

        loss = 0
        eps = 1e-6
        views = len(features)
        for view in range(views):
            anchor_feature = features[view]
            exp_pos_sim = 0
            exp_all_sim = 0
            for con in range(views):
                view_tmp = (view + con) % len(features)
                contrast_feature = features[view_tmp]
                anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
                # for numerical stability
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
                exp_sim = torch.exp(anchor_dot_contrast)
                pos_sim = exp_sim * mask
                all_sim = exp_sim * logits_mask
                exp_pos_sim += pos_sim.sum(1)
                exp_all_sim += all_sim.sum(1)
            # print('==================================exp_pos_sim\n', exp_pos_sim)
            # print('==================================exp_neg_sim\n', exp_neg_sim)

            l_tmp = torch.log(exp_pos_sim + eps) - torch.log(exp_all_sim + eps)
            # print('===========>l_tmp\n', l_tmp)
            loss += l_tmp.sum(0)
        loss = (self.temperature / self.base_temperature) * loss / (batch_size * views)
        return -loss


class TLCLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, reweight_epoch=-1, reweight_factor=0.05, annealing=500, tau=0.54):
        super(TLCLoss, self).__init__()
        self.reweight_epoch = reweight_epoch

        m_list = 1. / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False).cuda()
        self.m_list = m_list

        if reweight_epoch != -1:
            idx = 1
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights_enabled = None
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)

        # save diversity per_cls_weights
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                              requires_grad=False).to("cuda:0")
        self.T = (reweight_epoch + annealing) / reweight_factor
        self.tau = tau

    def to(self, device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, x, y):
        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, y.data.view(-1, 1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - 30 * batch_m
        return torch.exp(torch.where(index, x_m, x))

    def forward(self, outputs, y, logits, wight, epoch):
        loss = 0
        for i in range(outputs.shape[0] - 1):
            alpha = self.get_final_output(logits[i], y)
            S = alpha[i].sum(dim=1, keepdim=True)
            l = F.nll_loss(torch.log(alpha[i]) - torch.log(S), y, weight=self.per_cls_weights_base, reduction="none")

            # KL
            yi = F.one_hot(y, num_classes=alpha[i].shape[1])

            # adjusted parameters of D(p|alpha)
            alpha_tilde = yi + (1 - yi) * (alpha[i] + 1)
            S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
            kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(alpha_tilde.shape[1])) - torch.lgamma(
                alpha_tilde).sum(dim=1, keepdim=True) \
                 + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1, keepdim=True)
            l += epoch / self.T * kl.squeeze(-1)

            # diversity
            if self.per_cls_weights_diversity is not None:
                diversity_temperature = self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = 1
                temperature_mean = 1
            output_dist = F.log_softmax(logits[i] / diversity_temperature, dim=1)
            with torch.no_grad():
                mean_output_dist = F.softmax(outputs / diversity_temperature, dim=1)
            l -= 0.01 * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist,
                                                                       reduction="none").sum(dim=1)

            # dynamic engagement
            w = wight[i] / wight[i].max()
            w = torch.where(w > self.tau, True, False)
            l = (w * l).sum() / w.sum()
            loss += l.mean()

        return loss
