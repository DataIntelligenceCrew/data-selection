#!/bin/sh

python3 ../data-selection/main.py \
                --filepath "/localdisk3/data-selection/cifar-10-vectors-alexnet" \
                --threshold 0.9 \
                --number_of_partitions 10 \
                --data_generation 1 \
                --coverage_factor 30 \
                --sample_weight 1 \
                --composable 1 \
                --distribution_req 1000