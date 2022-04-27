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
    # for larger index, search in batches 
    batch_size = 1000
    for i in range(0, xb.shape[0], batch_size):
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], params.coverage_threshold)
        # print(i)
        try:
            for j in range(batch_size):
                # print(j)
                pl = set(I[limits[j] : limits[j+1]])
                posting_list[i+j] = pl
        except IndexError:
            break
    return posting_list

def write_posting_lists(params, posting_list_data, model_name):
    location = POSTING_LIST_LOC_GROUP.format(params.dataset, params.coverage_threshold, params.partitions, model_name)
    os.makedirs(location, exist_ok=True)
    for i in range(params.partitions):
        # posting_list_file = location + 'posting_list_' + str(i) + '.txt'
        posting_list_file = location + 'posting_list.txt'
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
    
    params.dataset = 'lfw'

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
    
    pl = get_full_data_posting_list(params, 'resnet-18')



