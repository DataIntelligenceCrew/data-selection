#!/bin/sh

for DIST in 50 100 200 300 500 700 900 
do
    for CF in 30
    do
        for ALG in 'MAB' 
        do
            python3 ../utils_models.py --dataset 'fashion-mnist' \
                                --coverage_threshold 0.9 \
                                --partitions 10 \
                                --algo_type $ALG \
                                --coverage_factor $CF \
                                --distribution_req $DIST
        done
    done
done
