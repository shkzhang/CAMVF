# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: QMUF
# @Author      : Shukang Zhang  
# @Owner       : fusheng
# @Data        : 2024/2/21
# @Time        : 17:37
# @Description :
import numpy as np
from sklearn.metrics import confusion_matrix


def sensitivity_specificity_mcc( y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return sensitivity, specificity,mcc

