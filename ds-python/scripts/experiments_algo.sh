#!/bin/sh


for DIST_REQ in 50 100 200 300 400 500 600 700 800 900   
do
    for ALG in 'k_centers_group'
    do
        python3 ../main_algo.py \
                --coverage_threshold 0.9 \
                --partitions 10 \
                --algo_type $ALG \
                --coverage_factor 30 \
                --distribution_req $DIST_REQ 
    done
done
