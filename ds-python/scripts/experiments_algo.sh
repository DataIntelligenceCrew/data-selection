#!/bin/sh


for DIST_REQ in 900 1000
do
    for ALG in 'greedyC' 'greedyNC'
    do
        python3 ../main_algo.py \
                --dataset 'cifar10' \
                --coverage_threshold 0.9 \
                --partitions 10 \
                --algo_type $ALG \
                --coverage_factor 30 \
                --distribution_req $DIST_REQ
    done
done