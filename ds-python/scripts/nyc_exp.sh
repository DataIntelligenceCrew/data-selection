#!/bin/sh



for CF in 5 10 20 
do 
    for DR in 50 100 200 300 400 500
    do 
        python3 ../nyc_coreset.py \
                --coverage_factor $CF \
                --distribution_req $DR 
    done 
done
