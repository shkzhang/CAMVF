# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: QMUF
# @Author      : Shukang Zhang
# @Owner       : fusheng
# @Data        : 2024/1/19
# @Time        : 12:06
# @Description : train utils
import logging
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm

from config.train_config import config
from model.cca import CCABase, CCAModel
from model.ctmc import CTMC
from model.ecamvf import ECAMVF
from model.etmc import ETMC
from model.qmf import rank_loss, QMF
from model.camvf import CAMVF
from utils.construct_data import get_dataloader
from utils.tools import AverageMeter, History, setup_seed
import argparse
import warnings
from sklearn.model_selection import KFold

args = None

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_args(_args):
    global args
    args = _args


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


def run():
    print(args)
    test_name = 'task-' + args.task_id + "-" + args.dataset + "-" + args.model + '-' + '_'.join(
        args.name_list) + '-' + str(args.batch_size)
    res_dir = './result/' + test_name + '/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    cpkg_path = os.path.join(res_dir, 'cpkg')
    if not os.path.exists(cpkg_path):
        os.makedirs(cpkg_path)
    loss_list = []
    acc_list = []
    seed_list = []
    model_type = args.train_type

    for i in args.seed:
        seed = i
        log_path = os.path.join(res_dir, f'{test_name}-Base.log')
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        setup_seed(seed)
        if args.dataset_type == 'tvt':
            train_loader, val_loader,test_loader = get_dataloader(args.root, args.name_list, args.batch_size, seed,
                                                                  args.dataset_type,
                                                                   model=model_type)
        elif args.dataset_type == 'tt':
            train_loader, test_loader = get_dataloader(args.root, args.name_list, args.batch_size, seed,
                                                                   args.dataset_type,
                                                                   model=model_type)
            val_loader = test_loader
        # train_loader, test_loader = get_dataloader(args.root, args.name_list, args.batch_size, seed, 'tt',model=model_type)
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

        # optimizer = optim.Adam(model.parameters(), lr=args.lr,eps=args.eps,weight_decay=args.weight_decay,betas=args.betas)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model.cuda()
        step_loss_list = []
        best_acc = 0
        tqdm_train = tqdm(range(args.epochs), desc=f'Train In Seed {seed}')
        model_path = os.path.join(cpkg_path, f'seed-{seed}-last.pth.tar')
        best_model = None
        for epoch in tqdm_train:
            loss_meter = train(model, optimizer, train_loader, epoch, model_type)
            step_loss_list.append(loss_meter.avg)
            val_loss, val_acc, val_view_acc = valid(model, val_loader, args.lambda_epochs, views=len(args.name_list),
                                                    type=model_type)
            # print(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model
                test_loss, test_acc, test_view_acc = valid(best_model, test_loader,
                                                           args.lambda_epochs, views=len(args.name_list), type=model_type)
                tqdm_train.set_postfix(val_best_acc="{:.6f}".format(best_acc),test_acc="{:.6f}".format(test_acc))

                torch.save({
                    'seed': seed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_size': args.batch_size,
                }, model_path)

        cpkg_dict = torch.load(model_path)
        model.load_state_dict(cpkg_dict['model_state_dict'])
        test_loss, test_acc, test_view_acc = valid(best_model, test_loader, args.lambda_epochs,
                                                   views=len(args.name_list),
                                                   type=model_type)
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        seed_list.append(seed)
        print(test_view_acc)

        logging.info(
            'Method:{:s}, Seed:{:3d}, Ave_loss:{:.3f}, Test_Acc: {:.4f}'.format(test_name, seed,
                                                                                test_loss, test_acc))
        print(('Method:{:s}, Seed:{:3d}, Ave_loss:{:.3f}, Test_Acc: {:.4f}'.format(test_name, seed,
                                                                                   test_loss,
                                                                                   test_acc)))

        csv_path = os.path.join(res_dir, f'{test_name}.csv')
        result_data = {'Seed': seed_list, 'Loss': loss_list, 'Acc': acc_list}
        df = pd.DataFrame(result_data)
        df.to_csv(csv_path, index=False)

    return np.mean(acc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='gcca',
                        choices=['ecamvf','camvf', 'tmc', 'etmc', 'qmf', 'gcca', 'dcca', 'kcca', 'dccae'],
                        metavar='LR',
                        help='config')
    parser.add_argument('--configs',
                        type=list,
                        # default=['ecamvf', 'ctmc', 'etmc', 'qmf', 'gcca', 'dcca', 'kcca', 'dccae'],

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
    else:
        default_config = config['default'].copy()
        args = argparse.Namespace(**default_config)
