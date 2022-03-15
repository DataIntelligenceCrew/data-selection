#!/bin/sh

for DIST_REQ in 100 
do
    for COMPOSABLE in 0 1
    do
        python3 ../models/model_data.py \
                --coverage_factor 30 \
                --sample_weight 1 \
                --composable $COMPOSABLE \
                --distribution_req $DIST_REQ
    done
done