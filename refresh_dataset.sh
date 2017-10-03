#!/bin/bash
# Script to refresh dataset or to obtain dataset if it is not present

if [ ! -d data/.git ]; then
    echo "Dataset is not found or broken."
    echo "Downloading fresh data from github..."
    rm -rf data
    git clone https://github.com/thanhtoantnt/precond-inference
    mv precond-inference data
fi

cd data && git pull
# if git pull fail
if [ $? -eq 0 ]; then
    echo "Dataset is ready"
else
    echo "Something went wrong. Please debug manually."
fi