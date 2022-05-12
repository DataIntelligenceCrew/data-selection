#!/bin/sh

for DIST in 50 100 200 300 400 500 600 700 800 900
do
    for CF in 30
    do
        for ALG in 'k_centersNC'
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
