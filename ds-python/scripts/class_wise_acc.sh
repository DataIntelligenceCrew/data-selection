#!/bin/sh

for DIST_REQ in 50 100 200 300 400 500 600 700 800 900
do
    for ALG in 'greedyC_group' 'greedyC_random' 'random' 'greedyNC' 'MAB'
    do
        python3 ../main_model.py --distribution_req $DIST_REQ --algo_type $ALG
    done
done