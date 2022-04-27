#!/bin/sh

for DR in 1000 2000 3000
do 
    python3 ../gender_detection.py --distribution_req $DR
done