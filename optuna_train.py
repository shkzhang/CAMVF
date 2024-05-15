# -*- coding:utf-8 -*-
# @Time    : 2023/12/8 22:17
# @Author  : Shukang Zhang
# @FileName: train.py
# @Software: PyCharm
# @Description: this is a program related to
import parser
import sys
import os

from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import Subset, DataLoader

import argparse
import logging
import os
import random
import warnings
from collections import Counter

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from config.train_config import config
from model.camvf import CAMVF
from model.cca import CCAModel, CCABase
from model.ctmc import CTMC
from model.ecamvf import ECAMVF
from model.etmc import ETMC
from model.qmf import QMF, rank_loss
from utils.construct_data import get_dataloader
from utils.tools import AverageMeter, History

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = None

def train(model, optimizer, train_loader, epoch, type='base'):
    model.train()
    loss_meter = AverageMeter()
    num_batches = len(train_loader)
    if type == 'cca':

        for _, data in enumerate(train_loader):
            target = Variable(data['label'].cuda())

            transform_data = model.cca_model.transform(data["views"])
            transform_data = torch.cat(transform_data, dim=1)
            transform_data = transform_data.cuda()
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(transform_data, target, epoch, 'train')
            # compute gradients and take step

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)
            optimizer.step()
            loss_meter.update(loss.item())
    elif type == 'qmf':
        histories = dict()
        for view in range(len(args.dims)):
            histories[view] = History(len(train_loader.dataset.dataset))

        for batch_idx, (data, target, idx) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, clf_loss, conf = model(data, target, 'train')
            rank_loss_value = 0
            for view in range(len(args.dims)):
                view_loss = nn.CrossEntropyLoss(reduction='none')(evidences[view], target).detach()
                histories[view].correctness_update(idx, view_loss, conf[view].squeeze())
                rank_loss_value += rank_loss(conf[view], idx, histories[view])
            loss = clf_loss + rank_loss_value

            # compute gradients and take step
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)
            optimizer.step()
            loss_meter.update(loss.item())

    else:
        for batch_idx, (data, target, _) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(data, target, epoch, 'train')
            # compute gradients and take step

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)
            optimizer.step()
            loss_meter.update(loss.item())
    return loss_meter



@torch.no_grad()
def valid(model, test_loader, epoch, views, type='base'):
    model.eval()
    loss_meter = AverageMeter()
    num_batches = len(test_loader)
    correct_num, data_num = 0, 0
    correct_view_num = dict()

    if type == 'cca':
        for _, data in enumerate(test_loader):
            target = Variable(data['label'].cuda())
            data_num += target.size(0)

            with torch.no_grad():
                transform_data = model.cca_model.transform(data["views"])
                transform_data = torch.cat(transform_data, dim=1)
                transform_data = transform_data.cuda()
                evidence, evidence_a, loss = model(transform_data, target, epoch, 'test')
                _, predicted = torch.max(evidence_a.data, 1)
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())

        return loss_meter.avg, correct_num / data_num, correct_view_num

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
                    evidence, evidence_a, loss, _ = model(data, target, 'train')
                else:
                    evidence, evidence_a, loss = model(data, target, epoch, 'test')
                _, predicted = torch.max(evidence_a.data, 1)
                correct_num += (predicted == target).sum().item()
                for view in range(views):
                    evidence_max, predicted_view = torch.max(evidence[view].data, 1)
                    correct_view_num[view] += (predicted_view == target).sum().item()
                loss_meter.update(loss.item())
        for view in range(views):
            correct_view_num[view] /= data_num
        return loss_meter.avg, correct_num / data_num, correct_view_num





def objective(trial):

    # Hyperparameters to be tuned by Optuna.
    epochs = trial.suggest_int('epochs', 400, 1000)
    lambda_epochs = trial.suggest_int('lambda_epochs', 0, 1000)
    beta = trial.suggest_float('beta', 0, 1)
    exp_eta = trial.suggest_float('exp_eta', 0, 1)
    batch_size = trial.suggest_int('batch_size', 10, 40)
    lr = trial.suggest_float('lr', 1e-5, 1e-3)
    alpha = trial.suggest_float('alpha', 0.01, 0.99)
    # seed = trial.suggest_int('seed', 0, 1000)
    args.alpha = alpha
    args.seed = 0
    args.folds = 5
    args.epochs = epochs
    args.batch_size = batch_size
    args.lambda_epochs = lambda_epochs
    args.beta = beta
    args.exp_eta = exp_eta

    # test_name = 'inter+intra+bs=' + str(args.batch_size)
    test_name = 'potuna+bs=' + str(args.batch_size)
    res_dir = './fold/' + test_name + '/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    fold_list = []
    fold_last_loss_list = []
    fold_last_acc_list = []
    fold = 0
    model_path = os.path.join(res_dir, f'seed-{args.seed}-best.pth.tar')
    model_type = args.train_type
    for train_loader,val_loader,test_loader in get_dataloader(args.root, args.name_list, batch_size, args.seed, 'k'):
        fold += 1
        best_acc = 0
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
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        model.cuda()
        epoch_tqdm = tqdm(range(args.epochs))
        epoch_tqdm.set_description(f"Fold {fold}")
        for epoch in epoch_tqdm:
            train(model, optimizer, train_loader, epoch)
            val_loss, val_acc, val_view_acc = valid(model, val_loader, args.lambda_epochs, views=len(args.name_list),
                                                    type=model_type)
            if val_acc>best_acc:
                best_acc = val_acc
                torch.save({
                    'seed': args.seed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_size': args.batch_size,
                }, model_path)
        cpkg_dict = torch.load(model_path)
        model.load_state_dict(cpkg_dict['model_state_dict'])
        val_loss, val_acc, val_view_acc = valid(model, test_loader, args.lambda_epochs, views=len(args.name_list),
                                                type=model_type)
        fold_list.append(fold)
        fold_last_loss_list.append(val_loss)
        fold_last_acc_list.append(val_acc)

    # test
    test_loss, test_acc = np.mean(fold_last_loss_list),np.mean(fold_last_acc_list)
    print(('=============>Method:{:s}, Seed:{:3d}, Test_ave_loss:{:.3f}, Test_Acc: {:.4f}'.format(test_name, args.seed,
                                                                                                  test_loss, test_acc)))
    csv_path = os.path.join(res_dir,
                            f'epochs-{args.epochs}-bs-{args.batch_size}-lr-{args.lr}-alpha-{args.alpha}-seed-{args.seed}-fold-{fold}.csv')
    result_data = {'Fold': fold_list, 'Fold_Last_Loss': fold_last_loss_list, 'Fold_Last_Acc': fold_last_acc_list}
    df = pd.DataFrame(result_data)
    df.to_csv(csv_path, index=False)
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='ecamvf',
                        choices=['ecamvf','camvf', 'tmc', 'etmc', 'qmf', 'gcca', 'dcca', 'kcca', 'dccae'],
                        metavar='LR',
                        help='config')
    args = parser.parse_args()
    assert args.config in config
    default_config = config['default'].copy()
    default_config.update(config[args.config])
    args = argparse.Namespace(**default_config)
    storage_name = "sqlite:///{}.db".format('study_ctmc_c')
    study = optuna.create_study(study_name='study_ctmc_c-kk-excel',
                                direction='maximize',
                                storage=storage_name,
                                load_if_exists=True)

    study.optimize(objective, n_trials=500,n_jobs=1)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    logging.info(study.best_params)
    logging.info(study.best_trial)
    logging.info(study.best_trial.value)

