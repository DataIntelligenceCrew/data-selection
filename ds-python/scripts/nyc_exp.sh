#!/bin/sh



for DR in 10 20 30 40 50
do 
    python3 ../main_algo.py \
            --coverage_factor $DR
done