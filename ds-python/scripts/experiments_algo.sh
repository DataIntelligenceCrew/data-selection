#!/bin/sh


for K in  1946 2791 4046 5452 8597 12805 16132 
# for DIST in 50 100 200 300 500 700 900 400 600 800
do
    for CF in 30
    do
        for ALG in 'k_centersNC' 
        do
            python3 ../main_algo.py \
                    --coverage_threshold 0.9 \
                    --partitions 10 \
                    --algo_type $ALG \
                    --coverage_factor $CF \
                    --k $K \
                    --dataset 'fashion-mnist' 
        done
    done
done
