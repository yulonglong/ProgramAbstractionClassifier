#!/bin/bash

THEANO_FLAGS="device=gpu0,mode=FAST_RUN,floatX=float32" python main.py -tr data/foo_100k.out -dt foo -o output -t SVC --epochs 50 -asm 200 -asb 2048

