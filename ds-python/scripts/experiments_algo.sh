#!/bin/sh


# for K in  1946 2791 4046 5452 8597 12805 16132 
for DIST in 50 100 200 300 400 500 600 700 800 900
do
    for C in 0.99 0.95 0.85 0.8 0.75
    do
        for ALG in 'two_phase_union'
        do
            python3 ../driver_code.py \
                    --coverage_threshold $C \
                    --partitions 10 \
                    --algo_type $ALG \
                    --coverage_factor 30 \
                    --distribution_req $DIST \
                    --dataset 'cifar10' 
        done
    done
done
