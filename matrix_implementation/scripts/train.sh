#!/bin/sh
for DIST_REQ in 5 10 20 40 80
do
    python3 ../models/main.py --sample_weight 0.1 \
                    --coreset 0 \
                    --train 0 \
                    --composable 0 \
                    --coverage_factor 30 \
                    --distribution_req $DIST_REQ
done