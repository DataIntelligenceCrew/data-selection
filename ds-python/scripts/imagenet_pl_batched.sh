#!/bin/bash
file='../imagenet_indices_1000.txt'

while read line
do 
    python3 ../utils_algo.py \
            --start_index $line
done < "$file"