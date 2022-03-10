#!/bin/sh


for DIST_REQ in 5 10 20 40 80 
do
    for COMPOSABLE in 0 1
    do
        python3 ../data-selection/main.py \
                --filepath "/localdisk3/data-selection/cifar-10-vectors-alexnet" \
                --threshold 0.9 \
                --number_of_partitions 10 \
                --data_generation 1 \
                --coverage_factor 30 \
                --sample_weight 0.1 \
                --composable $COMPOSABLE \
                --distribution_req $DIST_REQ
    done
done
