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

def convertStringToNumSequence(str):
    sequence = []
    maxValue = 0
    for i in xrange(len(str)):
        currIndex = ord(str[i]) + 1
        sequence.append(currIndex)
        if (maxValue < currIndex):
            maxValue = currIndex
    return maxValue, sequence

def read_dataset(args, model='default'):
    training_instances = []
    x = []
    y = []
    maxIndex = 0
    maxLength = 0
    totalLength = 0
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

            ##############################################
            ## Beginning of complicated reader for variable length input
            ##############################################
            if (args.num_str_parameter > 0): # if the features has strings
                maxValue, sequenceArr = convertStringToNumSequence(tokens[0])
                if (maxIndex < maxValue):
                    maxIndex = maxValue

                # just getting some statistics about the length
                if (maxLength < len(sequenceArr)):
                    maxLength = len(sequenceArr)
                totalLength += len(sequenceArr)

                # appending the array with zero or trimming the array to the correct size limit
                if (len(sequenceArr) < args.train_length_limit):
                    remainingLength = args.train_length_limit - len(sequenceArr)
                    emptyArr = [0]*remainingLength
                    sequenceArr = emptyArr + sequenceArr
                elif (len(sequenceArr) > args.train_length_limit):
                    sequenceArr = sequenceArr[:args.train_length_limit]
                assert (len(sequenceArr) == args.train_length_limit)

                floatArr = []
                for i in xrange(1, len(tokens)-1):
                    floatArr.append(float(tokens[i]))

                curr_x.append(sequenceArr)
                curr_x.append(floatArr)
            ##############################################
            ## End of complicated reader
            ##############################################

            else: # If the features are just real numbers
                for i in xrange(len(tokens)-1):
                    curr_x.append(float(tokens[i]))

            # logger.error(curr_x)

            x.append(curr_x)

            tokens[-1] = tokens[-1].strip()
            if tokens[-1] == 'T': y.append(1)
            else: y.append(0)

    logger.info("Maximum Index is : " + str(maxIndex))
    assert(maxIndex < args.vocab_size)
    logger.info("Maximum LENGTH is : " + str(maxLength))
    logger.info("Average LENGTH is : " + str(totalLength/len(x)))
    logger.info("LENGTH is trimmed and padded to : " + str(args.train_length_limit))

    return x, y
