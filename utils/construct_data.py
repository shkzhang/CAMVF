# -*- coding:utf-8 -*-
# @Time    : 2023/12/8 22:16
# @Author  : Shukang Zhang
# @FileName: construct_data.py
# @Software: PyCharm
# @Description: this is a program related to
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from torch.utils import data
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold


class MultiModalityData(data.Dataset):
    def __init__(self, root, name_list, seed, c_flag, flag, k=5, model='base'):
        self.root = root
        self.name_list = name_list
        self.data = dict()
        self.target = None
        self.model = model
        y = None
        X_scaler = StandardScaler()

        for i in range(len(self.name_list)):
            data_ = 'con_' + self.name_list[i] + '.csv'  # class 0
            data_path = os.path.join(self.root, data_)
            raw_data = pd.read_csv(data_path, header=0)
            raw_data = raw_data.dropna(axis=1,how='any')
            con_data = raw_data.iloc[:, :-1]
            if y is None:
                y = raw_data.iloc[:, -1].values
            mdata = con_data.values
            # mdata = X_scaler.fit_transform(mdata)

            # tt
            if c_flag == 'tt':
                X_train_i, X_test_i = train_test_split(mdata, test_size=0.2, random_state=seed)
                if flag == 'train':
                    self.data[i] = X_train_i
                if flag == 'test':
                    self.data[i] = X_test_i

            # tvt
            if c_flag == 'tvt':
                sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_index = None
                for train_index, test_index in sp.split(range(len(y)), y):
                    if flag == 'test':
                        self.target = y[test_index]
                        self.data[i] = mdata[test_index]
                        self.index = test_index

                    sp = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
                for v_train_index, valid_index in sp.split(train_index, y[train_index]):
                    if flag == 'train':
                        self.target = y[train_index[v_train_index]]
                        self.data[i] = mdata[train_index[v_train_index]]
                        self.index = train_index[v_train_index]

                    if flag == 'valid':
                        self.target = y[train_index[valid_index]]
                        self.data[i] = mdata[train_index[valid_index]]
                        self.index = train_index[valid_index]

            if c_flag == 'k':
                self.data[i] = mdata
                self.target = y

        if c_flag == 'tt':
            y_train, y_test = train_test_split(y, test_size=0.2, random_state=seed)
            if flag == 'train':
                self.target = y_train
                num_classes = max(self.target) + 1

                self.cls_num_list = np.histogram(self.target, bins=int(num_classes))[0].tolist()

            if flag == 'test':
                self.target = y_test



    def __getitem__(self, index):
        all_target = self.target[index]  # get the label related to index
        if self.model == 'cca':
            all_data = []
            for v_num in range(len(self.data)):
                all_data.append((self.data[v_num][index]).astype(np.float32))
            return {"views": all_data, "label": all_target, "index": index}
        all_data = dict()  # init data
        for v_num in range(len(self.data)):
            all_data[v_num] = (self.data[v_num][index]).astype(np.float32)  # get data related to index

        return all_data, all_target, index  # return data and label

    def __len__(self):
        return len(self.data[0])  # get the size of modalities


def get_dataloader(root, name_list, bs, seed=1, c_flag='tt', k=5, model='base'):
    assert c_flag in ['tvt', 'tt', 'k']
    if c_flag == 'tvt':
        train_loader = torch.utils.data.DataLoader(
            MultiModalityData(root, name_list, seed, c_flag, 'train', k=k, model=model), batch_size=bs, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            MultiModalityData(root, name_list, seed, c_flag, 'valid', k=k, model=model), batch_size=bs, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            MultiModalityData(root, name_list, seed, c_flag, 'test', k=k, model=model), batch_size=bs, shuffle=False)
        return train_loader, valid_loader, test_loader
    if c_flag == 'tt':
        train_loader = torch.utils.data.DataLoader(
            MultiModalityData(root, name_list, seed, c_flag, 'train', k=k, model=model), batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            MultiModalityData(root, name_list, seed, c_flag, 'test',k=k, model=model), batch_size=bs, shuffle=False)
        return train_loader, test_loader

    if c_flag == 'k':
        c_flag = 'tt'
        folder = []
        data = MultiModalityData(root, name_list, seed, c_flag, 'train', k, model)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(range(len(data.target)),data.target):
            sp = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            train_fold,val_fold = None,None
            for sp_train_index,val_index in sp.split(train_index, data.target[train_index]):
                train_fold = torch.utils.data.dataset.Subset(data, train_index[sp_train_index])
                val_fold = torch.utils.data.dataset.Subset(data, train_index[val_index])
            test_fold = torch.utils.data.dataset.Subset(data, test_index)
            folder.append((torch.utils.data.DataLoader(train_fold, batch_size=bs, shuffle=True),
                   torch.utils.data.DataLoader(val_fold, batch_size=bs, shuffle=True),
                   torch.utils.data.DataLoader(test_fold, batch_size=bs, shuffle=True)))
        return folder