#!/bin/sh
for DIST_REQ in 100 
do
    for COMPOSABLE in 0 
    do
        python3 ../models/main.py \
                --coreset 1 \
                --train 1 \
                --coverage_factor 30 \
                --sample_weight 1 \
                --composable $COMPOSABLE \
                --distribution_req $DIST_REQ
    done
done