#!/bin/sh

for D in 'mnist' 'cifar10'
do
    python3 ../main_model.py --dataset $D --coreset 0
done
