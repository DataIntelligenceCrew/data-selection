#!/bin/sh



for DR in 5 10 20 
do 
    python3 ../main_algo.py \
            --coverage_factor $DR
done