#!/bin/sh

for DIST in 200 300 400 500 600 700 800 900 100
do
    for CF in 0 30
    do
        for ALG in 'greedyNC' 'greedyC_random' 'k_centersNC' 'MAB'
        do
            python3 ../utils_models.py --dataset 'mnist' \
                                --coverage_threshold 0.9 \
                                --partitions 10 \
                                --algo_type $ALG \
                                --coverage_factor $CF \
                                --distribution_req $DIST
        done
    done
done