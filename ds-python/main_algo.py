'''
TODO: edit utils_algo.py to add more datasets and distribution requirements
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

LFW_LABELS = {'Asian' : 0, 
              'White' : 1, 
              'Black' : 2, 
              'Baby' : 3, 
              'Child' : 4, 
              'Youth' : 5, 
              'Middle Aged' : 6,
              'Senior' : 7, 
              'Indian' : 56
            }


def get_label_dict(dataset_name):
    labels_dict = dict()
    label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    labels = label_file.readlines()
    for l in labels:
        txt = l.split(':')
        key = int(txt[0].strip())
        value = int(txt[1].strip())
        if value not in labels_dict:
            labels_dict[value] = list()
        labels_dict[value].append(key)
    
    label_file.close()
    return labels_dict

def run_algo(params, dr=None):
    print('Running Algorithm: {0}\nDataset:{1}\nDistribution Requirement:{2}\nCoverage Factor:{3}\nCoverage Threshold:{4}\n'.format(
            params.algo_type,
            params.dataset,
            params.distribution_req,
            params.coverage_factor,
            params.coverage_threshold
        ))

    solution_data = []
    # TODO: add other methods
    if params.dataset != 'nyc_taxicab':
        labels = get_label_dict(params.dataset)
    if params.algo_type == 'greedyNC':
        if params.dataset.lower() == 'lfw':
            # dist_req = get_lfw_dr_config()
            # dist_req = [params.distribution_req] * params.num_classes
            dist_req = [0] * params.num_classes
            for idx in LFW_LABELS.values():
                dist_req[idx] = params.distribution_req

        else:
            dist_req = [params.distribution_req] * params.num_classes
        if dr:
            dist_req = dr
        if params.dataset == 'imagenet':
            posting_list = get_full_data_posting_list(params, params.model_type)
        elif params.dataset == 'nyc_taxicab':
            posting_list = get_posting_list_nyc_dist()
        else:
            posting_list = get_full_data_posting_list(params, params.model_type)
               
        s, cscore, res_time = greedyNC(params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold, posting_list, params.num_classes)
        solution_data.append((s, cscore, res_time))
    
    elif params.algo_type == 'stochastic_greedyNC':
        if params.dataset == 'lfw':
            # dist_req = get_lfw_dr_config()
            dist_req = [params.distribution_req] * params.num_classes
        else:
            dist_req = [params.distribution_req] * params.num_classes
        posting_list = get_full_data_posting_list(params, params.model_type)    
        s, cscore, res_time = stochastic_greedyNC(params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold, posting_list)
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
                args=(params.coverage_factor, dist_req, q, params.dataset, params.dataset_size, partition_data[i], params.model_type, params.coverage_threshold, params.num_classes)
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

    elif params.algo_type == 'greedyC_sample':
        dist_req = [math.ceil(params.distribution_req) / params.partitions] * params.num_classes
        full_data_posting_list = get_full_data_posting_list(params, params.model_type)
        partition_data = create_partitions_using_samples(params, full_data_posting_list, number_of_partitions=10)
        # spawn processes based on number of partitions
        # q = multiprocessing.Queue()
        # processes = []
        # for i in range(params.partitions):
        #     p = multiprocessing.Process(
        #         target=greedyC_random,
        #         args=(params.coverage_factor, dist_req, q, params.dataset, params.dataset_size, partition_data[i], params.model_type, params.coverage_threshold, params.num_classes)
        #     )
        #     processes.append(p)
        #     p.start()

        # # collect data from all processes
        # for p in processes:
        #     sol_data = q.get()
        #     solution_data.append(sol_data)

        # # delete all spawned processes
        # for p in processes:
        #     p.join()
    
    elif params.algo_type == 'random':
        if params.dataset == 'lfw':
            dist_req = [0] * params.num_classes
            for idx in LFW_LABELS.values():
                dist_req[idx] = params.distribution_req
            s, cscore, res_time = random_algo_lfw(dist_req)
        else:
            s, cscore, res_time = random_algo(labels, params.distribution_req)
        solution_data.append((s, cscore, res_time))


    elif params.algo_type == 'MAB':
        if params.dataset.lower() == 'lfw':
            # dist_req = get_lfw_dr_config()
            # dist_req = [params.distribution_req] * params.num_classes
            dist_req = [0] * params.num_classes
            for idx in LFW_LABELS.values():
                dist_req[idx] = params.distribution_req
        dist_req = [params.distribution_req] * params.num_classes
        posting_list = get_full_data_posting_list(params, params.model_type)
        s, cscore, res_time = bandit_algorithm(params.coverage_factor, dist_req, params.dataset, params.dataset_size, posting_list)
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
    elif params.algo_type == 'k_centers_group':
        dist_req = params.distribution_req
        q = multiprocessing.Queue()
        process = []
        for i in range(params.partitions):
            p = multiprocessing.Process(
                target=k_centers_group,
                args=(params.coverage_factor, dist_req, params.dataset, params.partitions, params.coverage_threshold, params.model_type, i, q)
            )
            process.append(p)
            p.start()
        
        for p in process:
            sol_data = q.get()
            solution_data.append(sol_data)
        
        for p in process:
            p.join()
    elif params.algo_type == 'k_centersNC' and params.k > 0:
        s, cscore, res_time = k_centersNC(params.k, params.dataset, params.dataset_size, params.model_type)
        solution_data.append((s, cscore, res_time))
    # calculate metrics
    coreset = set()
    cscores = 0
    response_time = 0
    for t in solution_data:
        coreset = coreset.union(t[0])
        cscores += t[1]
        response_time = max(response_time, t[2])
    
    print('Time Taken:{0}'.format(response_time))
    print('Solution Size: {0}'.format(len(coreset)))
    # report metrics
    metric_file_name = METRIC_FILE2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
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
    solution_file_name = SOLUTION_FILENAME2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
    with open(solution_file_name, 'w') as f:
        for s in coreset:
            f.write("{0}\n".format(s))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data selection parameters
    parser.add_argument('--dataset', type=str, default='nyc_taxicab', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=1, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyNC', help='which algorithm to use')
    parser.add_argument('--distribution_req', type=int, default=0, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=10, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used to produce the feature_vector')
    parser.add_argument('--k', type=int, default=0, help='number of centers for k_centersNC')
    params = parser.parse_args()
    
    if params.dataset == 'mnist':
        params.dataset_size = 60000
        params.num_classes = 10
    elif params.dataset == 'cifar10':
        params.num_classes = 10 
        params.dataset_size = 50000 
    elif params.dataset == 'fashion-mnist':
        params.num_classes = 10
        params.dataset_size = 60000
    elif params.dataset == 'cifar100':
        params.dataset_size = 50000
        params.num_classes = 100
    elif params.dataset == 'lfw':
        params.dataset_size = 13143
        params.num_classes = 72
    elif params.dataset == 'imagenet':
        params.dataset_size = 11060223
        params.num_classes = 10450
    elif params.dataset == 'nyc_taxicab':
        params.dataset_size = 68792
        params.num_classes = 1
    
    ## cifar10
    # k_to_dr = { 1077 : 50,  
    #             1743 : 100,
    #             3296 : 200, 
    #             4634 : 300, 
    #             5945 : 400, 
    #             7275 : 500, 
    #             8525 : 600, 
    #             9538 : 700,
    #             10788 : 800, 
    #             11985 : 900 }

    ## lfw
    # k_to_dr = { 153 : 50,  
    #             293 : 100,
    #             543 : 200, 
    #             769 : 300, 
    #             984 : 400, 
    #             1201 : 500, 
    #             1457 : 600, 
    #             1664 : 700,
    #             1882 : 800, 
    #             2086 : 900 }

    ## mnist
    # k_to_dr = {
    #     1132 : 50,
    #     1748 : 100,
    #     3150 : 200,
    #     4519 : 300,
    #     6111 : 400,
    #     8035 : 500,
    #     9876 : 600,
    #     11476 : 700,
    #     12907 : 800,
    #     14396 : 900
    # }

    # # fashion-mnist
    # k_to_dr = {
    #     1946 : 50,
    #     2791 : 100,
    #     4046 : 200,
    #     5452 : 300,
    #     8597 : 500,
    #     12805 : 700,
    #     16132 : 900
    # }
    # params.distribution_req = k_to_dr[params.k]
    run_algo(params)
    # distribution_req = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    # distribution_req = [0]
    
    # for i in distribution_req:
    #     params.distribution_req = i
    #     print('Running Algorithm:{0}\tDataset:{1}\tDR:{2}\tCF:{3}\tModel_FV:{4}\n'.format(
    #         params.algo_type,
    #         params.dataset,
    #         params.distribution_req,
    #         params.coverage_factor,
    #         params.model_type
    #     ))
    #     run_algo(params)
