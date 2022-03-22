#!/bin/sh
for DIST_REQ in 50 
do
    for COMPOSABLE in 0 1 
    do
        python3 ../models/main.py \
                --coreset 0 \
                --train 0 \
                --coverage_factor 30 \
                --sample_weight 1 \
                --composable $COMPOSABLE \
                --distribution_req $DIST_REQ
    done
done