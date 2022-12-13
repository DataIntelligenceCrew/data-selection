#!/bin/sh

for SEED in 1234 9876 5555 1111 2222 3333 4444 6666 1010 0000
do
    for DIST in 50 100 200 300 400 500 600 700 800 900
    do
        for ALG in 'MAB' 'greedyC_random'
        do
            for D in 'cifar10'
            do
                for LR in 0.02 0.01 0.001
                do
                    for C in 0.99 0.95 0.85 0.8 0.75
                    do 
                        python3 ../main_model.py --dataset $D \
                                            --partitions 10 \
                                            --algo_type $ALG \
                                            --coverage_factor 30 \
                                            --distribution_req $DIST \
                                            --seed $SEED \
                                            --lr $LR \
                                            --coverage_threshold $C
                    done
                done
            done
        done
    done
done



