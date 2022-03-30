#!/bin/sh

for DIST_REQ in 100 200 300 400 500 600 700 800 900
do 
    python3 ../main_model.py --distribution_req $DIST_REQ
done