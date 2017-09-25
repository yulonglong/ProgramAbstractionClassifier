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

def create_nn_model():
    from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, MaxPooling2D
    from keras.models import Model

    #Create your own input format (here 3x200x200)
    img_input = Input(shape=(1,16,8), name='image_input')

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=(1,16,8))(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Dropout(0.5)(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Dropout(0.5)(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dense(26, activation='softmax', name='predictions')(x)

    my_model = Model(input=img_input, output=x)
    my_model.summary()

    return my_model

def create_nn_model_signed():
    from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, MaxPooling2D
    from keras.models import Model

    #Create your own input format (here 3x200x200)
    inputs = Input(shape=(3,), name='inputs')
    x = Dense(32, activation='relu', name='fc1')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', name='fc2')(x)
    # x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', name='predictions')(x)

    my_model = Model(input=inputs, output=outputs)
    my_model.summary()

    return my_model

def run_model(dataset, model_type, args, out_dir=None, class_weight=None):
    import keras.utils.np_utils as np_utils

    ############################################################################################
    ## Set optimizers and compile NN model
    #

    import keras.optimizers as opt
    clipvalue = 0
    clipnorm = 10
    optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    loss = 'binary_crossentropy'
    metric = 'accuracy'

    model = create_nn_model_signed()
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    logger.info('model compilation completed!')

    ###############################################################################################################################
    ## Training
    #

    logger.info("Current dataset size: %i" % len(dataset))
    # Get the random dataset first

    train_x, train_y, dev_x, dev_y, test_x, test_y = H.getDatasetRandom(dataset, 6000, 2000, 2000)

    from Evaluator import Evaluator
    from keras.models import load_model
    
    ##############################################
    ## Active learning Loop
    #
    for i in xrange(1000):
        # Stop the active learning if the test set is larger than 5000
        if (len(test_y) > 5000):
            break

        if i > 0:
            logger.info("================ Active Loop %i ====================" % i)
            logger.info("Current dataset size: %i" % len(dataset))
            random.shuffle(dataset)

            activeData_x = np.empty([0,3])
            activeData_y = np.empty([0])

            for numIter in xrange(len(dataset)/args.active_sampling_batch_size):
                # Load current best model
                # model = load_model(args.out_dir_path + '/models/best_model_complete.h5')
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

            # Concatenate with the real dataset
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
            history = model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, verbose=0)
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
