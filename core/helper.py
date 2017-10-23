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
import random


logger = logging.getLogger(__name__)

def convertDataWithStrArgsSingle(x, y):
    new_train_x = []
    sequence_train_x = []
    float_train_x = []
    for i in xrange(len(x)):
        sequence_train_x.append(np.array(x[i][0]))
        float_train_x.append(np.array(x[i][1]))
    new_train_x.append(np.array(sequence_train_x))
    new_train_x.append(np.array(float_train_x))

    return new_train_x, np.array(y)

def convertDataWithStrArgs(train_x, train_y, valid_x, valid_y, test_x, test_y):
    new_train_x, new_train_y = convertDataWithStrArgsSingle(train_x, train_y)
    new_valid_x, new_valid_y = convertDataWithStrArgsSingle(valid_x, valid_y)
    new_test_x, new_test_y = convertDataWithStrArgsSingle(test_x, test_y)
    return new_train_x, new_train_y, new_valid_x, new_valid_y, new_test_x, new_test_y

def splitDatasetClass(dataset):
    dataset_list = [list(t) for t in zip(*dataset)]
    dataset_x = dataset_list[0]
    dataset_y = dataset_list[1]

    dataset_pos_x = []
    dataset_pos_y = []
    
    dataset_neg_x = []
    dataset_neg_y = []

    for i in xrange(len(dataset_y)):
        if dataset_y[i] == 1:
            dataset_pos_x.append(dataset_x[i])
            dataset_pos_y.append(dataset_y[i])
        elif dataset_y[i] == 0:
            dataset_neg_x.append(dataset_x[i])
            dataset_neg_y.append(dataset_y[i])

    assert (len(dataset_pos_x) == len(dataset_pos_y))
    assert (len(dataset_neg_x) == len(dataset_neg_y))

    logger.info("Dataset_pos size: %d" % len(dataset_pos_x))
    logger.info("Dataset_neg size: %d" % len(dataset_neg_x))

    return zip(dataset_pos_x, dataset_pos_y), zip(dataset_neg_x, dataset_neg_y)

def getDatasetRandomSingleClass(dataset, numTrain, numValid, numTest):
    """
    Method to initialize and split training, validation, and test set
    Random initialization
    Sampling without replacement, note that dataset variable is modified
    """
    random.shuffle(dataset)
    train = dataset[:numTrain]
    train_list = [list(t) for t in zip(*train)]
    train_x = train_list[0]
    train_y = train_list[1]
    del dataset[:numTrain]

    random.shuffle(dataset)
    valid = dataset[:numValid]
    valid_list = [list(t) for t in zip(*valid)]
    valid_x = valid_list[0]
    valid_y = valid_list[1]
    del dataset[:numValid]

    random.shuffle(dataset)
    test = dataset[:numTest]
    test_list = [list(t) for t in zip(*test)]
    test_x = test_list[0]
    test_y = test_list[1]
    del dataset[:numTest]

    return train_x, train_y, valid_x, valid_y, test_x, test_y
    # return np.array(train_x), np.array(train_y), np.array(valid_x), np.array(valid_y), np.array(test_x), np.array(test_y)

def getDatasetRandom(dataset_pos, dataset_neg, numTrain, numValid, numTest):
    train_pos_x, train_pos_y, dev_pos_x, dev_pos_y, test_pos_x, test_pos_y = getDatasetRandomSingleClass(dataset_pos, numTrain / 2, numValid / 2, numTest / 2)
    train_neg_x, train_neg_y, dev_neg_x, dev_neg_y, test_neg_x, test_neg_y = getDatasetRandomSingleClass(dataset_neg, numTrain / 2, numValid / 2, numTest / 2)
    
    train_x = np.concatenate((train_pos_x, train_neg_x),axis=0)
    train_y = np.concatenate((train_pos_y, train_neg_y),axis=0)
    dev_x = np.concatenate((dev_pos_x, dev_neg_x),axis=0)
    dev_y = np.concatenate((dev_pos_y, dev_neg_y),axis=0)
    test_x = np.concatenate((test_pos_x, test_neg_x),axis=0)
    test_y = np.concatenate((test_pos_y, test_neg_y),axis=0)

    return train_x, train_y, valid_x, valid_y, test_x, test_y
    
def getSubDataset(dataset, numIter, batchSize):
    subdataset = dataset[numIter*batchSize:(numIter+1)*batchSize]
    test_list = [list(t) for t in zip(*subdataset)]
    test_x = test_list[0]
    test_y = test_list[1]
    return np.array(test_x), np.array(test_y)

def removeFromDataset(indices, dataset):
    for index in sorted(indices, reverse=True):
        del dataset[index]

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

def compute_class_weight(train_y):
    """
    Compute class weight given imbalanced training data
    Usually used in the neural network model to augment the loss function (weighted loss function)
    Favouring/giving more weights to the rare classes.
    """
    import sklearn.utils.class_weight as scikit_class_weight

    class_list = list(set(train_y))
    class_weight_value = scikit_class_weight.compute_class_weight('balanced', class_list, train_y)
    class_weight = dict()

    # Initialize all classes in the dictionary with weight 1
    curr_max = np.max(class_list)
    for i in range(curr_max):
        class_weight[i] = 1

    # Build the dictionary using the weight obtained the scikit function
    for i in range(len(class_list)):
        class_weight[class_list[i]] = class_weight_value[i]

    logger.info('Class weight dictionary: ' + str(class_weight))
    return class_weight
