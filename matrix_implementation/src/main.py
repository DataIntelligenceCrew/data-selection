import data_generator
import os
from os.path import isfile, join
import random
import numpy as np
import multiprocessing
import argparse
import time
import algo

def main(args):
    # generate the posting lists 
    if args.data_generation == 0:
        generator = data_generator.MetadataGenerator(args.filepath, args.number_of_partitions, args.sample_weight, sample=True)
        generator.generate_metadata(args.threshold)

    # start the algorithm for each partition
    solution = set()
    dist_req = [5] * 10
    start_time = time.time()
    # solution = solution.union(algo.algorithm(0, args.coverage_factor, dist_req, args.sample_weight))
    for i in range(args.number_of_partitions):
        solution = solution.union(algo.algorithm(i, args.coverage_factor, dist_req, args.sample_weight))
    end_time = time.time()
    print("Solution Size: ", len(solution))
    print("Time taken: ", (end_time - start_time))

    output_file_path = "../runs/" + str(args.coverage_factor) + "_" + str(args.sample_weight) + ".txt"
    with open(output_file_path, 'w') as f:
        for s in solution:
            f.write(str(s) + "\n")
        time_str = "Time taken: " + str(end_time - start_time)
        f.write(time_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=False, default="")
    parser.add_argument('--threshold', type=float, required=False, default=0.9)
    parser.add_argument('--number_of_partitions', type=int, required=True, default=10)
    parser.add_argument('--data_generation', type=int, required=False, default=1)
    parser.add_argument('--coverage_factor', type=int, required=True, default=25)
    parser.add_argument('--sample_weight', type=float, required=True, default=0.1)
    args = parser.parse_args()
    main(args)