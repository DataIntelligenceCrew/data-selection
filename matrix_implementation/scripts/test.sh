#!/bin/sh

python3 ../models/main.py --sample_weight 0.1 \
                 --coreset 0 \
                 --train 1 \
                 --composable 1 \
                 --coreset_solution 25_0.1_1.txt