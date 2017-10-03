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
import activelearning as AL

logger = logging.getLogger(__name__)

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

    train_x, train_y, dev_x, dev_y, test_x, test_y = H.getDatasetRandomSingleClass(dataset, args.test_size * 3, args.test_size, args.test_size)
    if (args.is_equal_distribution):
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
            train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y = None, None, None, None, None, None
            if (args.is_equal_distribution):
                (train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y) = AL.obtain_data_active_learning_equal_distribution(args, dataset_pos, dataset_neg)
            else:
                (train_active_x, train_active_y, dev_active_x, dev_active_y, test_active_x, test_active_y) = AL.obtain_data_active_learning(args, dataset)

            # Concatenate additional dataset from active learning with the real dataset
            train_x = np.concatenate((train_x, train_active_x),axis=0)
            train_y = np.concatenate((train_y, train_active_y),axis=0)
            dev_x = np.concatenate((dev_x, dev_active_x),axis=0)
            dev_y = np.concatenate((dev_y, dev_active_y),axis=0)
            test_x = np.concatenate((test_x, test_active_x),axis=0)
            test_y = np.concatenate((test_y, test_active_y),axis=0)

        ############################################################################################
        ## Compute class weight (where data is usually imbalanced)
        #
        class_weight = H.compute_class_weight(np.array(train_y, dtype='float32'))

        ###############################################
        ## Real Training Starts
        #
        AL.print_shape(train_x, train_y, dev_x, dev_y, test_x, test_y)

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
            history = model.fit(train_x, train_y, batch_size=args.batch_size, class_weight=class_weight, nb_epoch=1, shuffle=True, verbose=0)
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
