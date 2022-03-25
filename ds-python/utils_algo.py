'''
TODO: Returns the posting lists from faiss search based on the input partition
'''
import numpy as np
import random
import os
import faiss
from paths import *
from os.path import isfile, join
import pickle
from collections import defaultdict
import argparse
import time


def create_partitions(params, random_partition=False):
    label_file = open(LABELS_FILE_LOC.format(params.dataset), 'r')
    lines = label_file.readlines()
    labels_dict = dict()
    delta = set()
    for l in lines:
        txt = l.split(':')
        key = int(txt[1].strip())
        value = int(txt[0].strip())
        if key not in labels_dict:
            labels_dict[key] = list()
        labels_dict[key].append(value)
        delta.add(value)

    if random_partition:
        ys = list(delta)
        random.shuffle(ys)
        ylen = len(ys)
        size = int(ylen / params.partitions)
        partitions = [ys[0+size*i : size*(i+1)] for i in range(params.partitions)]
        leftover = ylen - size*params.partitions
        edge = size*params.partitions
        for i in range(leftover):
                partitions[i%params.partitions].append(ys[edge+i])
        return partitions
    else:
        paritions = [v for v in labels_dict.values()]
        return paritions
        


def get_posting_lists(params, partition_data):
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(params.dataset), 'rb'))
    fv_partitions = np.take(feature_vectors, partition_data, 0)
    d = fv_partitions.shape[1]
    N = fv_partitions.shape[0]
    posting_list = {}
    # TODO: can this be the reason we get all vectors close to each other
    xb = fv_partitions.astype('float32')
    xb[:, 0] += np.arange(N) / 1000
    faiss_index = faiss.IndexFlatL2(d)
    faiss.normalize_L2(x=xb)
    faiss_index.add(xb)
    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    limits, D, I = faiss_index.range_search(xb, params.coverage_threshold)
    for i in range(xb.shape[0]):
        posting_list[i] = set(I[limits[i] : limits[i+1]])

    partition_posting_list = {}
    for key, value in posting_list.items():
        partition_posting_list[partition_data[key]] = set(partition_data[v] for v in value)
    
    return partition_posting_list



def get_full_data_posting_list(params):
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(params.dataset), 'rb'))
    d = feature_vectors.shape[1]
    N = feature_vectors.shape[0]
    posting_list = {}
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000
    faiss_index = faiss.IndexFlatL2(d)
    faiss.normalize_L2(x=xb)
    faiss_index.add(xb)
    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    # for larger index, search in batches 
    batch_size = 1000
    for i in range(0, xb.shape[0], batch_size):
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], params.coverage_threshold)
        for j in range(batch_size):
            posting_list[i+j] = set(I[limits[j] : limits[j+1]])
    
    return posting_list

def write_posting_lists(params, posting_list_data):
    location = POSTING_LIST_LOC.format(params.dataset, params.coverage_threshold, params.partitions)
    for i in range(params.partitions):
        posting_list_file = location + 'posting_list_alexnet_' + str(i) + '.txt'
        with open(posting_list_file, 'w') as f:
            for key, value in posting_list_data[i].items():
                f.write(str(key) + ' : ' + str(value) + '\n')
        f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
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

    # partitions = create_partitions(params)
    # print(partitions)
    start_time = time.time()
    posting_list = get_full_data_posting_list(params)
    end_time = time.time()
    print('Time Taken to generate metadata: {0}'.format(end_time - start_time))




