#!/bin/sh
for DIST_REQ in 100 200 300 400 500 600 700 800 900 1000
do
    for COMP in 0 1
    do
        python3 ../test/test.py \
                    --coverage_factor 30 \
                    --sample_weight 1 \
                    --composable $COMP \
                    --distribution_req $DIST_REQ
    done
done