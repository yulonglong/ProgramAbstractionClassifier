from __future__ import print_function
import random
import codecs
import sys
import logging
import re
import glob
import numpy as np
import pickle as pk
import re # regex
import copy

logger = logging.getLogger(__name__)
        
def read_dataset(args, model='default'):
    training_instances = []
    x = []
    y = []
    counter = 0
    class_mapping_index = 0
    with codecs.open(args.train_path, mode='r', encoding='ISO-8859-1') as input_file:
        for line in input_file:
            if counter == 0:
                counter += 1
                continue

            curr_x = []

            tokens = re.split(',', line.rstrip())
            # logger.error(tokens)

            assert (tokens[0][0] == '(')
            tokens[0] = tokens[0][1:]
            assert (tokens[-2][-1] == ')')
            tokens[-2] = tokens[-2][:-1]
            for i in xrange(len(tokens)-1):
                if (i < args.num_str_parameter):
                    curr_x.append(tokens[i])
                else:
                    curr_x.append(float(tokens[i]))

            # logger.error(curr_x)

            x.append(curr_x)

            tokens[-1] = tokens[-1].strip()
            if tokens[-1] == 'T': y.append(1)
            else: y.append(0)

    return x, y
