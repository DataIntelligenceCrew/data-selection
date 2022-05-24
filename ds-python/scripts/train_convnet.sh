#!/bin/sh

for SEED in 1234 9876
do
    for DIST in 50 
    do
        for ALG in 'dc'
        do
            for D in 'cifar10' 
            do
                python3 ../main_model.py --dataset $D \
                                    --partitions 10 \
                                    --algo_type $ALG \
                                    --coverage_factor 30 \
                                    --distribution_req $DIST \
                                    --seed $SEED
            done
        done
    done
done



