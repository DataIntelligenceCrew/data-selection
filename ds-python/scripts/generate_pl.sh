#!/bin/sh




for D in 'cifar10' 'cifar100' 'mnist' 'fashionmnist'
do 
    for C in 0.95 0.9 0.85 0.8
    do 
        python3 ../utils_algo.py \
                --dataset $D \
                --coverage_threshold $C 
    done 
done