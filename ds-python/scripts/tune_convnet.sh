#!/bin/sh

for W in 32 64 128 256
do
    for D in 1 2 3 4
    do 
        for N in "none" "batchnorm" "layernorm" "instancenorm" "groupnorm"
        do 
            for A in "sigmoid" "relu" "leakyrelu"
            do
                for P in "none" "maxpooling" "avgpooling"
                do
                    for DIST_REQ in 50 
                    do
                        for ALG in 'MAB' 
                        do
                            python3 ../main_model.py \
                                    --dataset 'cifar10' \
                                    --partitions 10 \
                                    --algo_type $ALG \
                                    --coverage_factor 30 \
                                    --distribution_req $DIST_REQ \
                                    --net_width $W \
                                    --net_depth $D \
                                    --net_act $A \
                                    --net_norm $N \
                                    --net_pooling $P
                        done
                    done
                done
            done
        done
    done
done