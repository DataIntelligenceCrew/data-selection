#!/bin/sh


# for K in  1077 1743 3296 4634 5945 7275 8525 9538 10788 11985  
for DIST in 153 293 543 769 984 1201 1457 1664 1882 2086
do
    for CF in 30
    do
        for ALG in 'k_centersNC' 
        do
            python3 ../main_algo.py \
                    --coverage_threshold 0.9 \
                    --partitions 10 \
                    --algo_type $ALG \
                    --coverage_factor $CF \
                    --k $DIST \
                    --dataset 'lfw' 
        done
    done
done
