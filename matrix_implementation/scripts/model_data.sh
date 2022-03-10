#!/bin/sh
for DIST_REQ in 5 10 20 40 80
do

    python3 ../models/model_data.py --coverage_factor 30 \
                                    --sample_weight 0.1 \
                                    --composable 0 \
                                    --distribution_req $DIST_REQ
done