from paths import *
from os.path import isfile, join
import argparse
import numpy as np
import os 


# TODO: change teh coverage test for the full posting list for greedyNC

def get_coreset(params):
    coreset = set()
    solution_file_name = SOLUTION_FILENAME.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.fv_type)
    f = open(solution_file_name, 'r')
    lines = f.readlines()
    for line in lines:
        point = int(line.strip())
        coreset.add(point)
    f.close()
    return coreset

def check_for_zeros(CC):
    incorrect = 0
    for e in CC:
        if e > 0:
            incorrect += 1
    return incorrect

def get_delta(params):
    location = POSTING_LIST_LOC.format(params.dataset, params.coverage_threshold, 1)
    posting_list_filepath = location + 'posting_list_alexnet.txt'
    posting_list_file = open(posting_list_filepath, 'r')
    lines = posting_list_file.readlines()
    delta = set()
    for line in lines:
        pl = line.split(':')
        point = int(pl[0])
        delta.add(point)


def test_coverage(params, coreset):
    delta_size = params.dataset_size
    posting_list = dict()
    delta = set()
    if params.algo_type == 'greedyC_group':
        location = POSTING_LIST_LOC_GROUP.format(params.dataset, params.coverage_threshold, params.partitions, 'group2')
    else:
        location = POSTING_LIST_LOC.format(params.dataset, params.coverage_threshold, params.partitions)
    
    
    for i in range(params.partitions):
        if params.algo_type == 'greedyNC' or params.algo_type == 'MAB':
            posting_list_file = open(os.path.join(location, 'posting_list_alexnet.txt'), 'r')
        else:
            posting_list_file = open(os.path.join(location, 'posting_list_resnet_{0}.txt'.format(i)), 'r')    
        lines = posting_list_file.readlines()
        for line in lines:
            pl = line.split(':')
            key = int(pl[0])
            if key in coreset:
                value = pl[1].split(',')
                value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
                arr = np.zeros(delta_size)
                arr[value] = 1
                posting_list[key] = arr
            delta.add(key)
    
    # assert(len(delta) == (args.sample_weight * delta_size))
    coreset_CC = np.zeros(delta_size)
    coreset_CC[list(delta)] = params.coverage_factor
    for key, value in posting_list.items():
        coreset_CC = np.subtract(coreset_CC, value)

    coverage_satisfied = check_for_zeros(coreset_CC)
    if coverage_satisfied == 0:
        print("Coverage Test Passed.....")
    else:
        print("Coverage Test Failed.....Uncovered:{0}".format(coverage_satisfied))


def test_distribution_req(params, coreset):
    label_file = open(LABELS_FILE_LOC.format(params.dataset), 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    labels = label_file.readlines()
    dist_req = np.zeros(len(label_ids_to_name.keys()))
    dist_req.fill(params.distribution_req)

    for l in labels:
        txt = l.split(':')
        key = int(txt[0].strip())
        label = int(txt[1].strip())
        if key in coreset:
            arr = np.zeros(len(label_ids_to_name.keys()))
            arr[label] = 1
            dist_req = np.subtract(dist_req, arr)
    
    distribution_satisfied = check_for_zeros(dist_req)
    if distribution_satisfied == 0:
        print("Distribution Req Test Passed.....")
    else:
        print("Distribution Req Test Failed.....")


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # data selection parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyC_group', help='which algorithm to use [greedyNC, greedyC, MAB, random, herding, k_center, forgetting]')
    parser.add_argument('--distribution_req', type=int, default=50, help='number of samples ')
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

    params.fv_type = 'resnet'
    fname = SOLUTION_FILENAME.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.fv_type)
    print('Testing for {0}'.format(fname))
    coreset = get_coreset(params)
    # delta = get_delta(params)
    test_coverage(params, coreset)
    test_distribution_req(params, coreset)