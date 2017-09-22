from __future__ import print_function
import argparse
import logging
import numpy as np
import sys
import utils as U
import reader as R
import models as M
import pickle as pk
import os.path
import copy
from time import time

logger = logging.getLogger(__name__)

def getDevFoldDrawCards(x, y):
    train_x, train_y, dev_x, dev_y = [], [], [], []

    for i in range(len(x)):
        if i%20 == 0:
            dev_x.append(x[i])
            dev_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(dev_x), np.array(dev_y)

def getFoldDrawCards(fold, x, y):
    train_x, train_y, dev_x, dev_y, test_x, test_y = [], [], [], [], [], []
    validation_fold = fold+1
    if validation_fold > 9: validation_fold = 0
    for i in range(len(x)):
        if i%10 == fold:
            test_x.append(x[i])
            test_y.append(y[i])
        elif i%10 == validation_fold:
            dev_x.append(x[i])
            dev_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(dev_x), np.array(dev_y), np.array(test_x), np.array(test_y)

def getFoldBulkCut(fold, x, y):
    train_x, train_y, dev_x, dev_y, test_x, test_y = [], [], [], [], [], []
    thresholdValid = float(fold) * 0.1
    thresholdTest = thresholdValid + 0.1
    if (thresholdValid == 1.0):
        thresholdValid = 0.0
    if (thresholdTest == 1.0):
        thresholdTest = 0.0

    bottomValid = thresholdValid*len(x)
    topValid = (thresholdValid+0.1)*len(x)
    bottomTest = thresholdTest*len(x)
    topTest = (thresholdTest+0.1)*len(x)

    for i in range(len(x)):
        if (i < topValid) and (i >= bottomValid):
            dev_x.append(x[i])
            dev_y.append(y[i])
        elif (i < topTest) and (i >= bottomTest):
            test_x.append(x[i])
            test_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(train_y), np.array(dev_x), np.array(dev_y), np.array(test_x), np.array(test_y)

def calculate_confusion_matrix(y_gold, y_pred):
    """
    Calculate the confusion matrix 
    True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN)
    """
    y_actual = copy.deepcopy(y_gold)
    y_hat = copy.deepcopy(y_pred)
    y_actual = y_actual.tolist()
    y_hat = y_hat.tolist()

    tps, fps, fns, tns = 0, 0, 0, 0
    y_hat_len = len(y_hat)

    for i in range(y_hat_len):
        if y_actual[i] == 1:
            if y_hat[i] == 1: tps += 1
            elif y_hat[i] == 0: fns += 1
        elif y_actual[i] == 0:
            if y_hat[i] == 0: tns += 1
            elif y_hat[i] == 1: fps += 1
    
    return (tps, fps, fns, tns)

def calculate_performance(tps, fps, fns, tns):
    """
    Calculate the performance/evaluation metrics given the confusion matrix (TP, FP, FN, TN)
    The evaluation metrics are:
    Recall/Sensitivity, Precision, Specificity, F1-score, F0.5-score, F1-recall-specificity, F0.5-recall-specificity.
    """
    recall, precision, specificity, f1, f05, accuracy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # f1ss is f1 score of sensitivity(recall) and specificity
    # f05ss is f0.5 score of sensitivity(recall) and specificity
    f1ss, f05ss = 0.0, 0.0

    if (tps + fns > 0): recall = float(tps) / float(tps + fns)
    if (tps + fps > 0): precision = float(tps) / float(tps + fps)
    if (tns + fps > 0): specificity = float(tns) / float(tns + fps)
    if (recall + precision > 0): f1 = 2.0 * recall * precision / (recall + precision)
    if (recall + precision > 0): f05 = 1.25 * recall * precision / ((0.25 * precision) + recall)
    if (tps + tns + fns + fps > 0): accuracy = float(tps + tns) / float (tps + tns + fns + fps)

    if (recall + specificity > 0): f1ss = 2.0 * recall * specificity / (recall + specificity)
    if (recall + specificity > 0): f05ss = 1.25 * recall * specificity / ((0.25 * specificity) + recall)

    return (recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss)

def calculate_confusion_matrix_performance(y_gold, y_pred):
    """
    Calculate the confusion matrix and several evaluation metrics given ground truth and predicted class.
    They are:
    True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN)
    Recall/Sensitivity, Precision, Specificity, F1-score, F0.5-score, F1-recall-specificity, F0.5-recall-specificity.
    """
    (tps, fps, fns, tns) = calculate_confusion_matrix(y_gold, y_pred)
    (recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss) = calculate_performance(tps, fps, fns, tns)
    return (tps, fps, fns, tns, recall, precision, specificity, f1, f05, accuracy, f1ss, f05ss)

def get_binary_predictions(pred, threshold=0.5):
    """
    Convert real number predictions between 0.0 to 1.0 to binary predictions (either 0 or 1)
    Using 0.5 as its default threshold unless specified
    """
    binary_pred = copy.deepcopy(pred)
    high_indices = binary_pred >= threshold
    low_indices = binary_pred < threshold
    binary_pred[high_indices] = 1
    binary_pred[low_indices] = 0

    return binary_pred
