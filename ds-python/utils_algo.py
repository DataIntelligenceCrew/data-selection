'''
TODO: Store everything in a sqlite3 db and load from there to avoid confusion
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
import json
import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt

def create_partitions(params, lables, random_partition=False):
    labels_dict = lables
    delta = lables.keys()
    if random_partition:
        # random.seed(1234)
        partitions = dict()
        for i in range(params.dataset_size):
            part_id = random.randint(0, params.partitions - 1)
            if part_id not in partitions:
                partitions[part_id] = list()
            
            partitions[part_id].append(i)
        
        return partitions
    else:
        paritions = [v for v in labels_dict.values()]
        return paritions
        

def create_partitions_using_samples(params, full_data_posting_list, number_of_partitions=10):
    delta = full_data_posting_list.keys()
    l = number_of_partitions
    # init stage
    rho = 0.1
    posting_list_size_data = dict()
    for k,v in full_data_posting_list.items():
        posting_list_size_data[k] = len(v)
    # use K-centers to find the l points to start with 
    from algorithms import k_centersNC
    top_l, score, time = k_centersNC(l, params.dataset, params.dataset_size, params.model_type)
    
    # descend_posting_size_list = [key for key, value in sorted(posting_list_size_data.items(), key=lambda x: (-x[1], x[0]))]
    # top_l = descend_posting_size_list[:l + 1]
    top_l = list(top_l)
    for n in top_l:
        print("{0} : {1}".format(n, posting_list_size_data[n]))

    partitions = dict()
    already_added = set()
    for i in range(l):
        # partitions[i] = list()
        init_node = top_l[i]
        init_nb = set(random.sample(full_data_posting_list[i], int(rho * float(len(full_data_posting_list[i]))))).difference(already_added)
        partitions[i] = list(init_nb)
        partitions[i].append(init_node)
        already_added.add(init_node)
        already_added = already_added.union(init_nb)

    # sample stage
    gamma = 0.1
    jump_rate = 10000
    def gamma_scheduler(iteration):
        return (iteration) / (iteration + jump_rate)


    def sample_scorer(point_id, partition_id):
        nb_score = ((len(full_data_posting_list[point_id].intersection(set(partitions[partition_id])))) / (len(full_data_posting_list[point_id])))
        size_score = 1 / (len(partitions[partition_id]) + 1)
        return gamma * nb_score + (1 - gamma) * size_score

    gamma_trend = list()
    delta = list(set(delta).difference(already_added))
    for p, idx in enumerate(delta):
        # calculate which partition does it belong to.
        partition_scores = [sample_scorer(p, i) for i in range(l)]
        part_id = partition_scores.index(max(partition_scores))
        partitions[part_id].append(p)
        gamma_trend.append(gamma)
        # gamma = gamma_scheduler(idx)
    

    part_sizes = [len(partitions[part]) for part in partitions.keys()]
    print(part_sizes)
    x = np.arange(len(gamma_trend))
    plt.plot(x, gamma_trend, 'o-')
    plt.xlabel('iterations')
    plt.ylabel('gamma value')
    plt.title('Gamma Trend vs Iteration for Jump Rate={0}'.format(jump_rate))
    plt.tight_layout()
    plt.show()
    plt.savefig('./figures/gamma_trend_jump_rate_{0}.png'.format(jump_rate))
    plt.cla()
    plt.clf()
    return partitions


    



def get_posting_lists(params, partition_data, model_name):
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(params.dataset, model_name), 'rb'))
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

def get_full_data_posting_list_imagenet(params, model_name):
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(params.dataset, model_name), 'rb'))
    d = feature_vectors.shape[1]
    N = feature_vectors.shape[0]
    posting_list = dict()
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000

    Xt = np.random.random((1000000, d)).astype(np.float32)  # 10000 vectors for training
    # Param of PQ
    M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
    nbits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
    # Param of IVF
    nlist = 10000  # The number of cells (space partition). Typical value is sqrt(N)
    # Param of HNSW
    hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32

    # Setup
    quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
    faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
    faiss_index.verbose = True
    # Train
    faiss_index.train(Xt)

    # Add
    faiss_index.add(xb)

    # Search
    faiss_index.nprobe = 16  # Runtime param. The number of cells that are visited for search.

    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    # limits, D, I = faiss_index.range_search(xb, params.coverage_threshold)
    # for i in range(xb.shape[0]):
    #     posting_list[i] = set(I[limits[i] : limits[i+1]])
    # try this if the above doesn't work.
    batch_size = 1000
    for i in range(0, xb.shape[0], batch_size):
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], params.coverage_threshold)
        print(i)
        try:
            for j in range(batch_size):
                # print(j)
                pl = set(I[limits[j] : limits[j+1]])
                posting_list[i+j] = pl
        except IndexError:
            break
    posting_list_file = "/localdisk3/data-selection/data/metadata/imagenet/posting_list.txt"
    with open(posting_list_file, 'w') as f:
        for key, value in posting_list.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    f.close()
    return posting_list


def get_full_data_posting_list(params, model_name):
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(params.dataset, model_name), 'rb'))
    d = feature_vectors.shape[1]
    N = feature_vectors.shape[0]
    posting_list = dict()
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000
    faiss_index = faiss.IndexFlatL2(d)
    faiss.normalize_L2(x=xb)
    faiss_index.add(xb)
    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    # loc = "/localdisk3/data-selection/data/metadata/imagenet/faiss_index"
    # f = open(loc, 'wb')
    # f = open(loc, 'rb')
    # faiss_index = pickle.load(f)
    # f.close()
    # pickle.dump(faiss_index, f, protocol=4)
    # print('successfully saved faiss index')
    # print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    # limits, D, I = faiss_index.range_search(xb[0:3], params.coverage_threshold)
    # for j in range(2):
    #     pl = set(I[limits[j] : limits[j+1]])
    #     posting_list[0+j] = pl
    # print(posting_list)
    # for larger index, search in batches 
    batch_size = 1000
    for i in range(0, xb.shape[0], batch_size):
        # faiss_index = faiss.IndexFlatL2(d)
        # faiss.normalize_L2(x=xb)
        # faiss_index.add(xb)
        # print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], params.coverage_threshold)
        # faiss_index.reset()
        # print(i)
        try:
            for j in range(batch_size):
                # print(j)
                pl = set(I[limits[j] : limits[j+1]])
                posting_list[i+j] = pl
        except IndexError:
            break
    posting_list_file = POSTING_LIST_LOC.format(params.dataset, params.coverage_threshold, model_name)
    with open(posting_list_file, 'w') as f:
        for key, value in posting_list.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    f.close()
    return posting_list

def write_posting_lists(params, posting_list_data, model_name):
    location = POSTING_LIST_LOC_GROUP.format(params.dataset, params.coverage_threshold, params.partitions, model_name)
    os.makedirs(location, exist_ok=True)
    for i in range(params.partitions):
        posting_list_file = location + 'posting_list_' + str(i) + '.txt'
        # posting_list_file = location + 'posting_list.txt'
        with open(posting_list_file, 'w') as f:
            for key, value in posting_list_data[i].items():
                f.write(str(key) + ' : ' + str(value) + '\n')
        f.close()



def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)


def generate_from_db():
    # alexnet_feature_vectors = np.ones((params.dataset_size, 512))
    labels = dict()
    db_file = r"/localdisk3/data-selection/cifar.db"
    conn = create_connection(db_file)
    cur = conn.cursor()
    cur.execute("SELECT id, label, resnet FROM images")
    rows = cur.fetchall()
    for row in rows:
        id = row[0] - 1
        label = row[1]
        if label not in labels:
            labels[label] = list()
        labels[label].append(id)
        # alexnet_fv = np.fromstring(row[2], dtype='float')
        # alexnet_feature_vectors[id] = alexnet_fv

    # print(alexnet_feature_vectors.shape)
    return labels




def get_lfw_dr_config():
    location = './lfw_dr.json'
    f = open(location, 'r')
    attrib_data = dict(json.load(f))
    f.close()
    attrib_config = list()
    for key, value in attrib_data.items():
        attrib_config.append(value)
    return attrib_config
    
    




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    params = parser.parse_args()
    
    # params.dataset = 'lfw'

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
    elif params.dataset == 'lfw':
        params.dataset_size = 13143
        params.num_classes = 73
    # feature_vectors, labels = generate_from_db(params)
    # partitions = create_partitions(params, labels)
    # for key, value in MODELS.items():
    #     print('Generating metadata for model: {0}'.format(key))
    #     posting_list_data = [get_posting_lists(params, p, key) for p in partitions]
    #     # posting_list_data = [get_full_data_posting_list(params, key)]
    #     write_posting_lists(params, posting_list_data,key)

    # posting_list_data = [get_full_data_posting_list(params, 'resnet')]
    # write_posting_lists(params, posting_list_data, 'resent')
    # attrib_config = get_lfw_dr_config()
    # print(attrib_config)
    
    # label_file = open(LABELS_FILE_LOC.format(params.dataset), 'r')
    # lines = label_file.readlines()
    # labels = dict()

    # for l in lines:
    #     txt = l.split(":")
    #     point = int(txt[0].strip())
    #     label = int(txt[1].strip())
    #     if label not in labels:
    #         labels[label] = list()
        
    #     labels[label].append(point)
    

    # partitions = create_partitions(params, labels, random_partition=False)
    # posting_list_data = [get_posting_lists(params, p, 'resnet-18') for p in partitions]
    # write_posting_lists(params, posting_list_data, 'resnet-18')



    pl = get_full_data_posting_list(params, 'resnet-18')
    # write_posting_lists(params, pl, 'resnet-18')



