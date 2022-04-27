#!/bin/sh

for DR in 1000 2000 3000
do 
    python3 ../main_model.py --distribution_req $DR
done