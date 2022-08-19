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
                    python3 ../main_model.py \
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