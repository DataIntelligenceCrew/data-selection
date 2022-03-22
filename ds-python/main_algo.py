import math
import os
import numpy as np
import multiprocessing
import argparse
from algorithms import *
from paths import *


def run_algo(params):
    solution_data = []
    # TODO: add other methods
    if params.algo_type == 'greedyNC':
        dist_req = [params.distribution_req] * params.num_classes    
        s, cscore, res_time = greedyNC(params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold)
        solution_data.append((s, cscore, res_time))

    elif params.algo_type == 'greedyC':
        dist_req = [math.ceil(params.distribution_req) / params.partitions] * params.num_classes
        # spawn processes based on number of partitions
        q = multiprocessing.Queue()
        processes = []
        for i in range(params.partitions):
            p = multiprocessing.Process(
                target=greedyC,
                args=(i, params.coverage_factor, dist_req, q, params.dataset, params.partitions, params.dataset_size, params.coverage_threshold)
            )
            processes.append(p)
            p.start()

        # collect data from all processes
        for p in processes:
            sol_data = q.get()
            solution_data.append(sol_data)

        # delete all spawned processes
        for p in processes:
            p.join()
    
    elif params.algo_type == 'random':
        s, cscore, res_time = random_algo(params.dataset, params.distribution_req)
        solution_data.append((s, cscore, res_time))

        
    # calculate metrics
    coreset = set()
    cscores = 0
    response_time = 0
    for t in solution_data:
        coreset = coreset.union(t[0])
        cscores += t[1]
        response_time = max(response_time, t[2])
    
    # report metrics
    metric_file_name = METRIC_FILE.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type)
    with open(metric_file_name, 'w') as f:
        f.write(
            'Dataset Size: {0}\nCoreset Size: {1}\nCoverage Score: {2}\nTime Taken: {3}\n'.format(
                int(params.dataset_size),
                len(coreset),
                cscores,
                response_time
            )
        )
    f.close()

    # log coreset points
    solution_file_name = SOLUTION_FILENAME.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type)
    with open(solution_file_name, 'w') as f:
        for s in coreset:
            f.write("{0}\n".format(s))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data selection parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyNC', help='which algorithm to use [greedyNC, greedyC, MAB, random, herding, k_center, forgetting]')
    parser.add_argument('--distribution_req', type=int, default=100, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    params = parser.parse_args()
    
    if params.dataset == 'mnist':
        params.dataset_size = 60000
        params.num_classes = 10
    elif params.dataset == 'cifar10':
        params.num_classes = 10 
        params.dataset_size = 50000 
    elif params.dataset == 'fashionmnist':
        params.num_classes = 10
        params.dataset_size = 60000
    elif params.dataset == 'cifar100':
        params.dataset_size = 50000
        params.num_classes = 100
    
    print('Running Algorithm:{0}\tDataset:{1}\tDR:{2}\tCF:{3}\n'.format(
        params.algo_type,
        params.dataset,
        params.distribution_req,
        params.coverage_factor
    ))

    run_algo(params)