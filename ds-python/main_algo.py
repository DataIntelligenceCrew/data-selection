'''
TODO: time profiling for greedy algorithms. 
    [hypothesis: loading metadata takes about 10 minutes]
'''
import math
import os
import numpy as np
import multiprocessing
import argparse
import faiss
from algorithms import *
from paths import *
from utils_algo import *

def run_algo(params):
    solution_data = []
    # TODO: add other methods
    labels = generate_from_db()
    if params.algo_type == 'greedyNC':
        dist_req = [params.distribution_req] * params.num_classes
        posting_list = get_full_data_posting_list(params, params.model_type)    
        s, cscore, res_time = greedyNC(params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold, posting_list)
        solution_data.append((s, cscore, res_time))

    elif params.algo_type == 'greedyC_random':
        dist_req = [math.ceil(params.distribution_req) / params.partitions] * params.num_classes
        partition_data = create_partitions(params, labels, random_partition=True)
        # spawn processes based on number of partitions
        q = multiprocessing.Queue()
        processes = []
        for i in range(params.partitions):
            p = multiprocessing.Process(
                target=greedyC_random,
                args=(params.coverage_factor, dist_req, q, params.dataset, params.dataset_size, partition_data[i], params.model_type, params.coverage_threshold)
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

    elif params.algo_type == 'MAB':
        dist_req = [params.distribution_req] * params.num_classes 
        s, cscore, res_time = bandit_algorithm(params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold, params.model_type)
        solution_data.append((s, cscore, res_time))
    
    elif params.algo_type == 'greedyC_group':
        dist_req = params.distribution_req
        # spawn processes based on number of partitions
        q = multiprocessing.Queue()
        processes = []
        for i in range(params.partitions):
            p = multiprocessing.Process(
                target=greedyC_group,
                args=(i, params.coverage_factor, dist_req, q, params.dataset, params.partitions, params.dataset_size, params.coverage_threshold, params.model_type)
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
    # calculate metrics
    coreset = set()
    cscores = 0
    response_time = 0
    for t in solution_data:
        coreset = coreset.union(t[0])
        cscores += t[1]
        response_time = max(response_time, t[2])
    
    print('Time Taken:{0}'.format(response_time))
    # report metrics
    metric_file_name = METRIC_FILE.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.model_type)
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
    solution_file_name = SOLUTION_FILENAME.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.model_type)
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
    parser.add_argument('--algo_type', type=str, default='greedyC_group', help='which algorithm to use [greedyNC, greedyC, MAB, random, herding, k_center, forgetting]')
    parser.add_argument('--distribution_req', type=int, default=50, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet', help='model used to produce the feature_vector')
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

    #run_algo(params)
    
    run_algo(params)
    distribution_req = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    
    for i in distribution_req:
        params.distribution_req = i
        print('Running Algorithm:{0}\tDataset:{1}\tDR:{2}\tCF:{3}\tModel_FV:{4}\n'.format(
            params.algo_type,
            params.dataset,
            params.distribution_req,
            params.coverage_factor,
            params.model_type
        ))
        run_algo(params)
