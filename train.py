from __future__ import print_function
import argparse
import codecs
import logging
import numpy as np
import sys
import core.utils as U
import core.helper as H
import core.reader as R
import core.models as M
import pickle as pk
import os.path
from time import time

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train-path", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='LogisticRegression', help="Model type for classification")
parser.add_argument("-ts", "--test-path", dest="test_path", type=str, metavar='<str>', default=None, help="The path to the test set")
parser.add_argument("-o", "--out", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The output folder")
parser.add_argument("-dt", "--dataset-type", dest="dataset_type", type=str, metavar='<str>', required=True, help="The type of dataset.")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size for training")
parser.add_argument("-be", "--batch-size-eval", dest="batch_size_eval", type=int, metavar='<int>', default=256, help="Batch size for evaluation")
parser.add_argument("--active-sampling-batch", dest="active_sampling_batch_size", type=int, metavar='<int>', default=512, help="Number of samples checked per active learning iteration")
parser.add_argument("--active-sampling-minimum", dest="active_sampling_minimum_addition", type=int, metavar='<int>', default=40, help="Number of samples required to continue training instead of sampling")
parser.add_argument("--test-amount-limit", dest="test_amount_limit", type=int, metavar='<int>', default=10000, help="Number test samples before stopping active learning (i.e., stop adding more samples into test set)")
parser.add_argument("--num-parameter", dest="num_parameter", type=int, metavar='<int>', default=4, help="The number of parameters being passed into the function being analyzed.")
parser.add_argument("--test-size", dest="test_size", type=int, metavar='<int>', default=2000, help="The amount of test samples being drawn at random at the beginning.")

parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=10, help="Number of epochs for Neural Net")
parser.add_argument("--test", dest="is_test", action='store_true', help="Flag to indicate testing (default=False)")
parser.add_argument("--equal-distribution", dest="is_equal_distribution", action='store_true', help="Flag to indicate active learning equal distribution (default=False)")

args = parser.parse_args()
out_dir = args.out_dir_path

##############################################################
## Make directory and initialize some required variables
#

U.mkdir_p(out_dir)
U.mkdir_p(out_dir + '/data')
U.mkdir_p(out_dir + '/preds')
U.mkdir_p(out_dir + '/models')
U.mkdir_p(out_dir + '/models/best_weights')
U.set_logger(out_dir)
U.print_args(args)

if args.is_test and args.test_path == None:
    logger.error("Please enter the path to the file for testing!")
    exit()

x, y = R.read_dataset(args.train_path, model=args.model_type)
dataset = zip(x,y)

accuracy = M.run_model(args, dataset, out_dir=out_dir, class_weight='balanced')
