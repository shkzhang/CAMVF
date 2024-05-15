# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: QMUF
# @Author      : Shukang Zhang  
# @Owner       : fusheng
# @Data        : 2024/2/19
# @Time        : 18:46
# @Description :
import logging
import os
import random
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy import interp
from torch.autograd import Variable

from config.train_config import config
from model.cca import CCABase, CCAModel
from model.ctmc import CTMC
from model.ecamvf import ECAMVF
from model.etmc import ETMC
from model.qmf import rank_loss, QMF
from model.camvf import CAMVF
from utils import metrics_utils
from utils.construct_data import get_dataloader
from utils.tools import AverageMeter, History, setup_seed
import argparse
import warnings
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.nn as nn

args = None
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@torch.no_grad()
def evaluate(model, test_loader, epoch, views, output_path, type='base'):
    model.eval()
    loss_meter = AverageMeter()
    num_batches = len(test_loader)
    correct_num, data_num = 0, 0
    correct_view_num = dict()
    all_target = []
    all_output = []
    all_prediction = []
    if type == 'cca':
        for _, data in enumerate(test_loader):
            target = Variable(data['label'].cuda())
            data_num += target.size(0)
            with torch.no_grad():
                transform_data = model.cca_model.transform(data["views"])
                transform_data = torch.cat(transform_data, dim=1)
                transform_data = transform_data.cuda()
                evidence, evidence_a, loss = model(transform_data, target, epoch, 'test')
                evidence_a = nn.LogSoftmax(dim=-1)(evidence_a)
                all_target.append(target.cpu().numpy())
                all_output.append(evidence_a.cpu().numpy())
                _, predicted = torch.max(evidence_a.data, 1)
                all_prediction.append(predicted.cpu().numpy())
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())
    else:
        for view in range(views):
            correct_view_num[view] = 0
        for batch_idx, (data, target, _) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.long().cuda())
                if type == 'qmf':
                    evidence, evidence_a, loss, _ = model(data, target, 'test')
                else:
                    evidence, evidence_a, loss = model(data, target, epoch, 'test')
                evidence_a = nn.LogSoftmax(dim=-1)(evidence_a)
                all_target.append(target.cpu().numpy())
                all_output.append(evidence_a.cpu().numpy())
                _, predicted = torch.max(evidence_a.data, 1)
                all_prediction.append(predicted.cpu().numpy())
                correct_num += (predicted == target).sum().item()
                for view in range(views):
                    evidence_max, predicted_view = torch.max(evidence[view].data, 1)
                    correct_view_num[view] += (predicted_view == target).sum().item()
                loss_meter.update(loss.item())
        for view in range(views):
            correct_view_num[view] /= data_num

    all_target = np.concatenate(all_target)
    all_output = np.concatenate(all_output)
    all_prediction = np.concatenate(all_prediction)

    result = dict()
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = None
    class_id = 1
    fpr, tpr, threshold = metrics.roc_curve(all_target == class_id,
                                            all_output[:, class_id],
                                            pos_label=True)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(all_target, all_prediction)
    recall, f1, sen, spe, mcc = None, None, None, None, None
    f1 = metrics.f1_score(all_target, all_prediction)
    # recall = metrics.recall_score(all_target, all_prediction)
    sen,spe,mcc = metrics_utils.sensitivity_specificity_mcc(all_target, all_prediction)
    #
    # mse = nn.MSELoss()(torch.Tensor(all_prediction), torch.Tensor(all_target)).cpu()
    result = {'tpr': mean_tpr, 'fpr': mean_fpr, 'acc': acc, 'recall': recall, 'auc': auc, 'f1': f1, 'sen': sen,
              'spe': spe, 'mcc': mcc}
    return loss_meter.avg, correct_num / data_num, correct_view_num, result


def run():
    print(args)
    test_name = 'task-' + args.task_id + "-" + args.dataset + "-" + args.model + '-' + '_'.join(
        args.name_list) + '-' + str(args.batch_size)
    model_dir = './result/' + test_name + '/'
    res_dir = './result/evaluate/' + test_name + '/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    cpkg_path = os.path.join(model_dir, 'cpkg')
    if not os.path.exists(cpkg_path):
        os.makedirs(cpkg_path)
    loss_list = []
    acc_list = []
    seed_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    model_type = args.train_type
    mean_fpr = None
    mean_tprs = []
    result_list = []
    for i in args.seed:
        seed = i
        log_path = os.path.join(res_dir, f'{test_name}-Base.log')
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        setup_seed(seed)
        seed_view_acc = {}

        train_loader, test_loader = get_dataloader(args.root, args.name_list, args.batch_size, seed, 'tt',
                                                   model=model_type)
        model_path = os.path.join(cpkg_path, f'seed-{seed}-last.pth.tar')
        if not os.path.exists(model_path):
            continue
        cpkg_dict = torch.load(model_path)

        cca_model = None
        latent_dimensions = 16
        if 'cca' in args.model:
            latent_dimensions = args.latent_dimensions
            cca_model = CCAModel(latent_dimensions=latent_dimensions, dims=args.dims, model_type=args.model)
            cca_model.fit(args.epochs, train_loader, test_loader)
        model_dict = {
            'camvf': CAMVF(args.dims, args.classes, args.alpha, args.beta, args.lambda_epochs, args.encode_dim,
                          args.proj_dim, args.exp_eta),
            'ecamvf': ECAMVF(args.dims, args.classes, args.alpha, args.beta, args.lambda_epochs, args.encode_dim,
                            args.proj_dim, args.exp_eta),
            'qmf': QMF(args.dims, args.classes, args.alpha, args.lambda_epochs, args.encode_dim, args.proj_dim),
            'tmc': CTMC(args.dims, args.classes, args.alpha, args.beta, args.lambda_epochs, args.encode_dim,
                         args.proj_dim),
            'etmc': ETMC(args.dims, args.classes, args.alpha, args.beta, args.lambda_epochs, args.encode_dim,
                         args.proj_dim),
            'gcca': CCABase(cca_model, len(args.dims), args.classes, latent_dimensions, args.encode_dim,
                            args.proj_dim),
            'dcca': CCABase(cca_model, len(args.dims), args.classes, latent_dimensions, args.encode_dim,
                            args.proj_dim),
            'dccae': CCABase(cca_model, len(args.dims), args.classes, latent_dimensions, args.encode_dim,
                             args.proj_dim),
            'kcca': CCABase(cca_model, len(args.dims), args.classes, latent_dimensions, args.encode_dim,
                            args.proj_dim),
        }

        model = model_dict[args.model]
        model.cuda()

        model.load_state_dict(cpkg_dict['model_state_dict'])
        test_loss, test_acc, test_view_acc, result = evaluate(model, test_loader, args.lambda_epochs,
                                                              views=len(args.name_list),
                                                              output_path=res_dir,
                                                              type=model_type)

        mean_fpr = result['fpr']
        mean_tprs.append(result['tpr'])
        result_list.append(result)
        seed_view_acc = dict(Counter(seed_view_acc) + Counter(test_view_acc))

        logging.info(
            'Method:{:s}, Seed:{:3d}, Ave_loss:{:.3f}, Test_Acc: {:.4f}'
            .format(test_name, seed,
                    test_loss,
                    test_acc))
        print(('Method:{:s}, Seed:{:3d}, Ave_loss:{:.3f}, Test_Acc: {:.4f}'
               .format(test_name, seed,
                       test_loss,
                       test_acc)))
        loss_list.append(test_loss)
        acc_list.append(test_acc)
        seed_list.append(seed)


    mean_tprs = np.mean(mean_tprs, axis=0)
    data = {'1-Specificity': mean_fpr, 'Sensitivity': mean_tprs}
    roc_df = pd.DataFrame(data)
    roc_df.to_csv(os.path.join(res_dir, f'class-{0}-roc.csv'), index=False)

    print(os.path.join(res_dir, f'class-{0}-roc.csv'))

    print(('Method:{:s}, Ave_loss:{:.3f}, Test_Acc: {:.4f}'
           .format(test_name,
                   np.mean(loss_list),
                   np.mean(acc_list))))
    sum_values = {}
    count_values = {}
    dist_list={}

    for d in result_list:
        for key, value in d.items():
            if isinstance(value, float):
                if key in sum_values:
                    sum_values[key] += value
                    count_values[key] += 1
                    dist_list[key].append(value)
                else:
                    sum_values[key] = value
                    count_values[key] = 1
                    dist_list[key] = []
                    dist_list[key].append(value)

    # 计算平均值并更新字典
    for key in sum_values:
        sum_values[key] /= count_values[key]

    csv_path = os.path.join(res_dir, f'{test_name}.csv')
    result_data = {'Seed': seed_list, 'Loss': loss_list}
    result_data.update(dist_list)
    df = pd.DataFrame(result_data)
    df.to_csv(csv_path, index=False)

    # 打印结果
    print("Average values:")
    print(sum_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='gcca',
                        choices=['ecamvf','camvf', 'tmc', 'etmc', 'qmf', 'gcca', 'dcca', 'kcca', 'dccae'],
                        metavar='LR',
                        help='config')
    parser.add_argument('--configs', type=List,
                        # default=['qmuf', 'eqmuf', 'ctmc', 'etmc', 'qmf', 'gcca', 'dcca', 'kcca', 'dccae'],
                        metavar='LR',
                        help='configs')

    args = parser.parse_args()
    if args.configs is not None:
        for _config in args.configs:
            assert _config in config
            default_config = config['default'].copy()
            default_config.update(config[_config])
            args = argparse.Namespace(**default_config)
            run()
    elif args.config is not None:
        assert args.config in config
        default_config = config['default'].copy()
        default_config.update(config[args.config])
        args = argparse.Namespace(**default_config)
        run()
