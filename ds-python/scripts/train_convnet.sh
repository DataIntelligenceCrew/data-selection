#!/bin/sh

for DIST in 50 100 200 300 500 700 900 
do
    for CF in 30
    do
        for ALG in 'k_centersNC'
        do
            python3 ../main_model.py --dataset 'fashion-mnist' \
                                --partitions 10 \
                                --algo_type $ALG \
                                --coverage_factor $CF \
                                --distribution_req $DIST
        done
    done
done
