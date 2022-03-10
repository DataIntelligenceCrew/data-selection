"""
Main Driver code that handles data generation as well as solving the set cover
"""

import data_generator
import os
from os.path import isfile, join
import random
import numpy as np
import multiprocessing
import argparse
import time
import algorithm



def calculate_cscore(solution, posting_list, delta_size=50000):
    cmatrix = np.zeros(shape=(len(solution), delta_size))
    for idx, s in enumerate(solution):
        cmatrix[idx] = posting_list[s]
    
    new_matrix = np.dot(cmatrix.T, cmatrix)
    return np.trace(new_matrix)


def get_output_filename(cf, sp, dist, c):
    return str(cf) + "_" + str(sp) + "_" + str(dist) + "_" + str(c)

def main(args):
    # generate the posting lists 
    if args.data_generation == 0:
        generator = data_generator.MetadataGenerator(args.filepath, args.number_of_partitions, args.sample_weight, sample=True)
        generator.generate_metadata(args.threshold)
    
    solution_data = []
    dist_req = [args.distribution_req] * 10
    
    if args.composable == 0:
        # start the algorithm for each partition
        # multithreaded
        q = multiprocessing.Queue()
        processes = []
        for i in range(args.number_of_partitions):
            p = multiprocessing.Process(
                target=algorithm.composable_algorithm,
                args=(i, args.coverage_factor, dist_req, args.sample_weight, q)
            )
            processes.append(p)
            p.start()        

        for p in processes:
            sol_data = q.get()
            solution_data.append(sol_data)


        for p in processes:
            p.join()

    else:
        s, cscore, res_time = algorithm.algorithm(args.coverage_factor, dist_req, args.sample_weight)
        solution_data.append((s, cscore, res_time))


    solution = set()
    cscores = 0
    response_time = 0
    for tuple in solution_data:
        solution = solution.union(tuple[0])
        cscores += tuple[1]
        response_time = max(response_time, tuple[2])

    print("Solution Size: ", len(solution))
    print("Solution Coverage Score: ", cscores)
    print("Time taken: ", response_time)

    output_filename = get_output_filename(args.coverage_factor, args.sample_weight, args.distribution_req, args.composable)
    output_file_path = "../runs/" + output_filename +".txt"
    with open(output_file_path, 'w') as f:
        for s in solution:
            f.write(str(s) + "\n")
    
    f.close()

    metric_file_path = "../runs/metrics_" + output_filename + ".txt"
    with open(metric_file_path, 'w') as f:
        f.write(
            "Delta Size: {0}\nSolution Size: {1}\nCoverage Score: {2}\nTime Taken: {3}\nDistribution Requirement: {4}\nCoverage Factor: {5}\n".format(
                int(50000 * args.sample_weight), 
                len(solution), 
                cscores, 
                response_time, 
                args.distribution_req,
                args.coverage_factor
            )
        )
    f.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=False, default="")
    parser.add_argument('--threshold', type=float, required=False, default=0.9)
    parser.add_argument('--number_of_partitions', type=int, required=True, default=10)
    parser.add_argument('--data_generation', type=int, required=False, default=1)
    parser.add_argument('--coverage_factor', type=int, required=True, default=25)
    parser.add_argument('--sample_weight', type=float, required=True, default=0.1)
    parser.add_argument('--composable', type=int, required=True, default=1)
    parser.add_argument('--distribution_req', type=int, required=True, default=50)
    args = parser.parse_args()
    main(args)