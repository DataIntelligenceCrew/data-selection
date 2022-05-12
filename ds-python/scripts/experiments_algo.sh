#!/bin/sh


# for K in  1077 1743 3296 4634 5945 7275 8525 9538 10788 11985  
for DIST in 50 100 200 300 500 700 900 400 600 800
do
    for CF in 0 30
    do
        for ALG in 'k_centersNC' 'greedyNC' 'greedyC_random' 'MAB' 
        do
            python3 ../main_algo.py \
                    --coverage_threshold 0.9 \
                    --partitions 10 \
                    --algo_type $ALG \
                    --coverage_factor $CF \
                    --distribution_req $DIST \
                    --dataset 'fashion-mnist' 
        done
    done
done
