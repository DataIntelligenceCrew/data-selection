import math 
import os 
import multiprocessing
import argparse
import faiss
from methods import *
from algorithms import *
from paths import *
from utils_algo import *



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





def generate_coreset(params, dr=None):
    print('Running Algorithm: {0}\nDataset:{1}\nDistribution Requirement:{2}\nCoverage Factor:{3}\nCoverage Threshold:{4}\n'.format(
            params.algo_type,
            params.dataset,
            params.distribution_req,
            params.coverage_factor,
            params.coverage_threshold
        ))
    
    solution_data = []
    # TODO: add other methods
    labels = get_label_dict(params.dataset)

    if params.algo_type == 'gfkc':
        if params.dataset.lower() == 'lfw':
            # dist_req = get_lfw_dr_config()
            # dist_req = [params.distribution_req] * params.num_classes
            # dist_req = [0] * params.num_classes
            # for idx in LFW_LABELS.values():
            #     dist_req[idx] = params.distribution_req
            pass

        else:
            dist_req = [params.distribution_req] * params.num_classes
        if dr:
            dist_req = dr
        if params.dataset == 'imagenet':
            posting_list = get_full_data_posting_list(params, params.model_type)
        else:
            posting_list = get_full_data_posting_list(params, params.model_type)
        
        coreset, response_time = gfkc(params.coverage_factor, dist_req, params.dataset, params.dataset_size, posting_list, params.num_classes)
        solution_data.append((coreset, response_time))
    
    elif params.algo_type =='two_phase':
        if params.dataset.lower() == 'lfw':
            # dist_req = get_lfw_dr_config()
            # dist_req = [params.distribution_req] * params.num_classes
            # dist_req = [0] * params.num_classes
            # for idx in LFW_LABELS.values():
            #     dist_req[idx] = params.distribution_req
            pass

        else:
            dist_req = [0] * params.num_classes
        # if dr:
        #     dist_req = dr
        if params.dataset == 'imagenet':
            posting_list = get_full_data_posting_list(params, params.model_type)
        else:
            posting_list = get_full_data_posting_list(params, params.model_type)
        
        # coverage_coreset, response_time1 = gfkc(params.coverage_factor, dist_req, params.dataset, params.dataset_size, posting_list, params.num_classes)
        coverage_coreset, _ , response_time1 = greedyNC(params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold, posting_list, params.num_classes)
        if dr:
            dist_req = dr
        else:
            dist_req = [params.distribution_req] * params.num_classes
        
        # print(type(posting_list))
        # posting_list = get_full_data_posting_list(params, params.model_type)
        # print(type(posting_list))
        # print(type(coverage_coreset))
        coreset, response_time2 = two_phase(posting_list, coverage_coreset, params.coverage_factor, dist_req, params.dataset_size, params.dataset, params.num_classes)
        solution_data.append((coreset, response_time1 + response_time2))
    elif params.algo_type == 'cgfkc':
        dist_req = [math.ceil(params.distribution_req) / params.partitions] * params.num_classes
        partition_data = create_partitions(params, labels, random_partition=True)
        solution_queue = multiprocessing.Queue()
        threads = []
        for i in range(params.partitions):
            p = multiprocessing.Process(
                target=cgfkc,
                args=(solution_queue, params.coverage_factor, dist_req, params.dataset, params.dataset_size, params.coverage_threshold, params.model_type, params.num_classes, partition_data)
            )
        
        for p in threads:
            sol_data = solution_queue.get()
            solution_data.append(sol_data)
        
        for p in threads:
            p.join()
    
    # calculate metrics
    coreset = set()
    cscores = 0
    response_time = 0
    for t in solution_data:
        coreset = coreset.union(t[0])
        response_time = max(response_time, t[1])
    
    print('Time Taken:{0}'.format(response_time))
    print('Solution Size: {0}'.format(len(coreset)))
    # report metrics
    metric_file_name = METRIC_FILE2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
    with open(metric_file_name, 'w') as f:
        f.write(
            'Dataset Size: {0}\nCoreset Size: {1}\nTime Taken: {2}\n'.format(
                int(params.dataset_size),
                len(coreset),
                response_time
            )
        )
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data selection parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='two_phase', help='which algorithm to use')
    parser.add_argument('--distribution_req', type=int, default=20, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
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
    
    generate_coreset(params)