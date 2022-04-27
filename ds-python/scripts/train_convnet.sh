#!/bin/sh

for DIST_REQ in 50
do 
    python3 ../main_model.py --distribution_req $DIST_REQ
done