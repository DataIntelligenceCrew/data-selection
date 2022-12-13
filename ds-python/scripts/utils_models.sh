#!/bin/sh

for DIST in 50 100 200 300 400 500 600 700 800 900 
do
    for C in 0.99 0.95 0.85 0.8 0.75
    do
        for ALG in 'MAB' 'greedyC_random' 
        do
            python3 ../utils_models.py --dataset 'cifar10' \
                                --coverage_threshold $C \
                                --partitions 10 \
                                --algo_type $ALG \
                                --coverage_factor 30 \
                                --distribution_req $DIST
        done
    done
done
