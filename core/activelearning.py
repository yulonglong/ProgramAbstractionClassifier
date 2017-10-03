from __future__ import print_function
import random
import codecs
import sys
import logging
import re
import glob
import numpy as np
import pickle as pk
import helper as H
import re # regex
import copy
from time import time

from sklearn.metrics import accuracy_score

from Evaluator import Evaluator
from keras.models import load_model

logger = logging.getLogger(__name__)


def print_shape_subset(x, y):
    logger.info("x.shape : " + str(x.shape))
    logger.info("y.shape : " + str(y.shape))

def print_shape(train_x, train_y, dev_x, dev_y, test_x, test_y):
    logger.info("train_x.shape : " + str(train_x.shape))
    logger.info("train_y.shape : " + str(train_y.shape))
    logger.info("dev_x.shape   : " + str(dev_x.shape))
    logger.info("dev_y.shape   : " + str(dev_y.shape))
    logger.info("test_x.shape  : " + str(test_x.shape))
    logger.info("test_y.shape  : " + str(test_y.shape))

def obtain_data_active_learning_equal_distribution(args, dataset_pos, dataset_neg):
    logger.info("Current dataset_pos size: %i" % len(dataset_pos))
    logger.info("Current dataset_neg size: %i" % len(dataset_neg))
    random.shuffle(dataset_pos)
    random.shuffle(dataset_neg)

    # Load current best model (keras)
    model = load_model(args.out_dir_path + '/models/best_model_complete.h5')

    activeData_pos_x = np.empty([0,args.num_parameter])
    activeData_pos_y = np.empty([0])
    activeData_neg_x = np.empty([0,args.num_parameter])
    activeData_neg_y = np.empty([0])

    active_sampling_batch_size = args.active_sampling_batch_size / 2
    numIter = -1
    while ((numIter+1) * active_sampling_batch_size) < min(len(dataset_pos), len(dataset_neg)):
        numIter += 1
        # Get a subset of the data to evaluate with the current model
        currActiveData_pos_x, currActiveData_pos_y = H.getSubDataset(dataset_pos, numIter, active_sampling_batch_size)
        currActiveData_neg_x, currActiveData_neg_y = H.getSubDataset(dataset_neg, numIter, active_sampling_batch_size)

        # Obtain the prediction with the current dataset
        activePred_pos = model.predict(currActiveData_pos_x, batch_size=args.batch_size_eval).squeeze()
        activePred_neg = model.predict(currActiveData_neg_x, batch_size=args.batch_size_eval).squeeze()

        # Get the indices of the dataset where the prediction is between 0.4 and 0.6
        indices_pos = np.where(np.logical_and(activePred_pos>=0.4, activePred_pos<=0.6))[0]
        indices_neg = np.where(np.logical_and(activePred_neg>=0.4, activePred_neg<=0.6))[0]

        # logger.info("Indices to be removed:")
        # logger.info(indices)
        # logger.info("Corresponding scores:")
        # logger.info(activePred[indices])

        # Remove the dataset to be added to the training/validation/test set from the global dataset
        H.removeFromDataset(indices_pos,dataset_pos)
        H.removeFromDataset(indices_neg,dataset_neg)

        # Get the desired dataset with unsure probabilities
        currActiveData_pos_x = currActiveData_pos_x[indices_pos]
        currActiveData_pos_y = currActiveData_pos_y[indices_pos]
        currActiveData_neg_x = currActiveData_neg_x[indices_neg]
        currActiveData_neg_y = currActiveData_neg_y[indices_neg]

        # Combine with the global activeData
        activeData_pos_x = np.concatenate((activeData_pos_x, currActiveData_pos_x),axis=0)
        activeData_pos_y = np.concatenate((activeData_pos_y, currActiveData_pos_y),axis=0)
        activeData_neg_x = np.concatenate((activeData_neg_x, currActiveData_neg_x),axis=0)
        activeData_neg_y = np.concatenate((activeData_neg_y, currActiveData_neg_y),axis=0)
        logger.info("Current shape of data (pos and neg) to add to the main dataset:")
        print_shape_subset(activeData_pos_x, activeData_pos_y)
        print_shape_subset(activeData_neg_x, activeData_neg_y)

        assert (len(activeData_pos_x) == len(activeData_pos_y))
        assert (len(activeData_neg_x) == len(activeData_neg_y))

        # Get the minimum length of each class
        # Trim the dataset to the minimum size so they have equal size (equal distribution positive and negative)
        min_length = min(len(activeData_pos_x), len(activeData_neg_x))
        activeData_pos_x = activeData_pos_x[:min_length]
        activeData_pos_y = activeData_pos_y[:min_length]
        activeData_neg_x = activeData_neg_x[:min_length]
        activeData_neg_y = activeData_neg_y[:min_length]
        if min_length * 2 >= args.active_sampling_minimum_addition:
            break

    # Concatenate positive and negative into one array
    train_active_x = np.concatenate((activeData_pos_x[:len(activeData_pos_x)*3/5], activeData_neg_x[:len(activeData_neg_x)*3/5]),axis=0)
    train_active_y = np.concatenate((activeData_pos_y[:len(activeData_pos_y)*3/5], activeData_neg_y[:len(activeData_neg_y)*3/5]),axis=0)
    dev_active_x = np.concatenate((activeData_pos_x[len(activeData_pos_x)*3/5:len(activeData_pos_x)*4/5], activeData_neg_x[len(activeData_neg_x)*3/5:len(activeData_neg_x)*4/5]),axis=0)
    dev_active_y = np.concatenate((activeData_pos_y[len(activeData_pos_y)*3/5:len(activeData_pos_y)*4/5], activeData_neg_y[len(activeData_neg_y)*3/5:len(activeData_neg_y)*4/5]),axis=0)
    test_active_x = np.concatenate((activeData_pos_x[len(activeData_pos_x)*4/5:], activeData_neg_x[len(activeData_neg_x)*4/5:]),axis=0)
    test_active_y = np.concatenate((activeData_pos_y[len(activeData_pos_y)*4/5:], activeData_neg_y[len(activeData_neg_y)*4/5:]),axis=0)
    print_shape(train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y)
    # logger.info(train_active_x)
    # logger.info(dev_active_x)
    # logger.info(test_active_x)

    logger.info("Current dataset_pos size: %i" % len(dataset_pos))
    logger.info("Current dataset_neg size: %i" % len(dataset_neg))

    return (train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y)


def obtain_data_active_learning(args, dataset):
	logger.info("Current dataset size: %i" % len(dataset))
	random.shuffle(dataset)

	# Load current best model
	model = load_model(args.out_dir_path + '/models/best_model_complete.h5')

	activeData_x = np.empty([0, args.num_parameter])
	activeData_y = np.empty([0])

	for numIter in xrange(len(dataset)/args.active_sampling_batch_size):
		# Get a subset of the data to evaluate with the current model
		currActiveData_x, currActiveData_y = H.getSubDataset(dataset, numIter, args.active_sampling_batch_size)
		# Obtain the prediction with the current dataset
		activePred = model.predict(currActiveData_x, batch_size=args.batch_size_eval).squeeze()
		# Get the indices of the dataset where the prediction is between 0.4 and 0.6
		indices = np.where(np.logical_and(activePred>=0.4, activePred<=0.6))[0]

		# logger.info("Indices to be removed:")
		# logger.info(indices)
		# logger.info("Corresponding scores:")
		# logger.info(activePred[indices])

		# Remove the dataset to be added to the training/validation/test set from the global dataset
		H.removeFromDataset(indices,dataset)
		# Get the desired dataset with unsure probabilities
		currActiveData_x = currActiveData_x[indices]
		currActiveData_y = currActiveData_y[indices]
		
		# Combine with the real thing
		activeData_x = np.concatenate((activeData_x, currActiveData_x),axis=0)
		activeData_y = np.concatenate((activeData_y, currActiveData_y),axis=0)
		logger.info("Current shape of data to add to the main dataset:")
		print_shape_subset(activeData_x, activeData_y)

		if len(activeData_x) > args.active_sampling_minimum_addition:
			break

	train_active_x = activeData_x[:len(activeData_x)*3/5]
	train_active_y = activeData_y[:len(activeData_y)*3/5]
	dev_active_x = activeData_x[len(activeData_x)*3/5:len(activeData_x)*4/5]
	dev_active_y = activeData_y[len(activeData_y)*3/5:len(activeData_x)*4/5]
	test_active_x = activeData_x[len(activeData_x)*4/5:]
	test_active_y = activeData_y[len(activeData_x)*4/5:]
	print_shape(train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y)
	# logger.info(train_active_x)
	# logger.info(dev_active_x)
	# logger.info(test_active_x)

	logger.info("Current dataset size: %i" % len(dataset))

	return (train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y)
