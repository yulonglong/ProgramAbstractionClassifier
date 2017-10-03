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

####################################################################################################
#### BEGIN SECTION FOR NEURAL NET
##

def create_nn_model(args):
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model

    inputs = Input(shape=(args.num_parameter,), name='inputs')
    x = Dense(32, activation='relu', name='fc1')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', name='fc3')(x)
    outputs = Dense(1, activation='sigmoid', name='predictions')(x)

    my_model = Model(input=inputs, output=outputs)
    my_model.summary()

    return my_model

def obtain_data_active_learning(args, dataset_pos, dataset_neg):
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

def run_model(args, dataset, out_dir=None, class_weight=None):
    ############################################################################################
    ## Set optimizers and compile NN model
    #
    import keras.optimizers as opt
    clipvalue = 0
    clipnorm = 10
    optimizer = opt.Adamax(lr=0.001, clipnorm=clipnorm, clipvalue=clipvalue)
    loss = 'binary_crossentropy'
    metric = 'accuracy'

    model = create_nn_model(args)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    logger.info('model compilation completed!')

    ###############################################################################################################################
    ## Training
    #

    # Split dataset into positive and negative subset
    dataset_pos, dataset_neg = H.splitDatasetClass(dataset)
    # Get the random dataset first
    train_x, train_y, dev_x, dev_y, test_x, test_y = H.getDatasetRandom(dataset_pos, dataset_neg, args.test_size * 3, args.test_size, args.test_size)
    
    ##############################################
    ## Active learning Loop
    #

    counter = 0
    # Stop the active learning if the test set is larger than the specified amount
    while (len(test_y) < args.test_amount_limit):
        counter += 1
        if counter > 1:
            logger.info("================ Active Loop %i ====================" % counter)
            (train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y) = obtain_data_active_learning(args, dataset_pos, dataset_neg)
        
            # Concatenate additional dataset from active learning with the real dataset
            train_x = np.concatenate((train_x, train_active_x),axis=0)
            train_y = np.concatenate((train_y, train_active_y),axis=0)
            dev_x = np.concatenate((dev_x, dev_active_x),axis=0)
            dev_y = np.concatenate((dev_y, dev_active_y),axis=0)
            test_x = np.concatenate((test_x, test_active_x),axis=0)
            test_y = np.concatenate((test_y, test_active_y),axis=0)

        ###############################################
        ## Real Training Starts
        #
        print_shape(train_x, train_y, dev_x, dev_y, test_x, test_y)

        evl = Evaluator(
            out_dir,
            (train_x, train_y),
            (dev_x, dev_y),
            (test_x, test_y),
            no_threshold=True
        )
        
        logger.info('---------------------------------------------------------------------------------------')
        logger.info('Initial Evaluation:')
        evl.evaluate(model, -1)

        # Print and send email Init LSTM
        content = evl.print_info()

        total_train_time = 0
        total_eval_time = 0

        for ii in range(args.epochs):
            t0 = time()
            history = model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, shuffle=True, verbose=0)
            tr_time = time() - t0
            total_train_time += tr_time

            # Evaluate
            t0 = time()
            best_acc = evl.evaluate(model, ii)
            evl_time = time() - t0
            total_eval_time += evl_time

            logger.info('Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm)' % (ii, tr_time, tr_time/60.0, evl_time, evl_time/60.0))
            logger.info('[Train] loss: %.4f , metric: %.4f' % (history.history['loss'][0], history.history['acc'][0]))

            # Print and send email Epoch LSTM
            content = evl.print_info()
