#!/bin/sh

for D in 'fashion-mnist'
do
    python3 ../main_model.py --dataset $D --coreset 0
done
