#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz && rm cifar-10-python.tar.gz
echo "downloaded the data and putting it in: " $DATADIR
