#!/bin/sh


for DIST_REQ in 50 100 200 300 400 500
do
    python3 ../main_algo.py \
            --dataset 'cifar10' \
            --coverage_threshold 0.9 \
            --partitions 10 \
            --algo_type 'greedyNC' \
            --coverage_factor 30 \
            --distribution_req $DIST_REQ 
done
