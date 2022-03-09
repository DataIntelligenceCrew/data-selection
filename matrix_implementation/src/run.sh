#!/bin/sh

python3 main.py --filepath "/localdisk3/data-selection/cifar-10-vectors-alexnet" \
                --threshold 0.9 \
                --number_of_partitions 10 \
                --data_generation 0 \
                --coverage_factor 25 \
                --sample_weight 0.1