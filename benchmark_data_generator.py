from collections import defaultdict
import random
import os
from os.path import isfile, join
import sys
import argparse
import numpy as np
import faiss
from pprint import pprint
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import spatial
import math
import pickle
# Randomly generates N d-dimensional vectors and assigns each to a class out of K randomly
# Stores them in a sql database 


class FaissSearch:
    def __init__(self, feature_vectors, seeded=False):
        self.d = feature_vectors.shape[1]
        self.N = feature_vectors.shape[0]
        self.posting_list = {}
        self.inverted_index = {}
        self.class_labels = {}
        if seeded:
            np.random.seed(1234)
        self.xb = feature_vectors.astype('float32')
        self.xb[:, 0] += np.arange(self.N) / 1000.
        self.cpu_index = faiss.IndexFlatL2(self.d)
        faiss.normalize_L2(x=self.xb)
        self.cpu_index.add(self.xb)
        print("successfully built faiss index")
        print("Index size: ", self.cpu_index.ntotal)
    
    def threshold_search(self, threshold, gpu=False):
        start_time = time.time()
        # think about batching the search queries for faster result
        # think about PQ codes to parameterize the vectors 
        # batching the dataset into multiple index and then combining results
        if gpu:
            limits, D, I = self.threshold_search_gpu(threshold)
        else:
            # search for the first 10 and so on... 
            self.batched_threshold_search(threshold)
            # limits, D, I = self.cpu_index.range_search(self.xb, threshold)
        for i in range(self.xb.shape[0]):
            print("result_size for query {0} is {1}".format(i, len(self.posting_list[i])))
            # self.posting_list[i] = set(I[limits[i] : limits[i+1]])
            # self.class_labels[i] = random.randint(1, 10)
            for e in self.posting_list[i]:
                if e not in self.inverted_index:
                    self.inverted_index[e] = list()
                self.inverted_index[e].append(i)
        print("Time taken to search and generate metadata: %s" % (time.time() - start_time))
        posting_list_file = '/localdisk3/data-selection/posting_list.txt'
        inverted_index_file = '/localdisk3/data-selection/inverted_index.txt'
        class_label_file = '/localdisk3/data-selection/class_label.txt'
        self.write_to_file(self.posting_list, posting_list_file)
        self.write_to_file(self.inverted_index, inverted_index_file)
        self.write_to_file(self.class_labels, class_label_file)
    
    def batched_threshold_search(self, threshold):
        batch_size = 1000
        for i in range(0, self.xb.shape[0], batch_size):
            limits, D, I = self.cpu_index.range_search(self.xb[i:i+batch_size], threshold)
            for j in range(batch_size):
                self.posting_list[i+j] = set(I[limits[j] : limits[j+1]])
                self.class_labels[i+j] = random.randint(1, 10)



    # range_search on gpu using k-nn search
    def threshold_search_gpu(self, threshold):
        init_k = 100
        max_k = 2048
        # send cpu_index to gpu
        gpu_ids = "0,1,2,3"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        self.gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        
        ii = defaultdict(list)
        dd = defaultdict(list)

        k = init_k
        D, I = self.gpu_index.search(self.xb, k)
        n = len(D)
        r, c = np.where(D <= threshold)
        actual_r = r

        while True:
            for row, col, act_r in zip(r, c, actual_r):
                ii[act_r].append(I[row, col])
                dd[act_r].append(D[row, col])

                continue_idx = [rr for rr, v in dd.items() if (len(v) == k)]

                if len(continue_idx) == 0:
                    break

                k *= 2
                if k >= max_k:
                    break
                
                D, I = self.gpu_index.search(self.xb[continue_idx], k=k)

                prev_k = int(k/2)
                D = D[:, prev_k:]
                I = I[:, prev_k:]
                r, c = np.where(D <= threshold)
                _, cts = np.unique(r, return_counts=True)
                actual_r = np.repeat(continue_idx, cts)
        
        sorted_rows = range(n)
        lims = np.concatenate([np.array([0]), np.cumsum([len(dd[i]) for i in sorted_rows])]).astype(np.uint64)
        D = np.array([sl for l in [dd[r] for r in sorted_rows] for sl in l])
        I = np.array([sl for l in [ii[r] for r in sorted_rows] for sl in l])

        return lims, D, I
        

    def write_to_file(self, d, filepath):
        with open(filepath, 'w') as f:
            for key, value in d.items():
                f.write(str(key) + " : " + str(value) + "\n")
        f.close()

def create_dataset(args):
    d = args.dimension               # dimension
    nb = args.N                      # database size
    nq = args.N                      # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    d_time = time.time()
    D = create_distance_matrix(xb, xb, args)
    print("Time taken for D: %s" % (time.time() - d_time))
    posting_list = {}
    for i in range(len(D)):
        for j in range(len(D[0])):
            if D[i][j] <= 0.75:
                if i not in posting_list:
                    posting_list[i] = set()
                posting_list[i].add(j)
    
    for key, value in posting_list.items():
        print("Posting List of Query {0} is {1}".format(key, len(value)))

    
    gpu_ids = "0,1"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    xb[:, 0] += np.arange(nb) / 1000.
    cpu_index = faiss.IndexFlatL2(d)   # build the index
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    # gpu_index.add(xb)
    # print(index.is_trained)
    # for normal L2 distance for higher dimensions the search is just the points it self.
    # after normalizing cosine similarity is better
    faiss.normalize_L2(x=xb)     # uncomment if cosine similarity is needed
    cpu_index.add(xb)                  # add vectors to the index
    print("Dataset size: ", cpu_index.ntotal)
    start_time = time.time()
    limits, D, I = cpu_index.range_search(xb, thresh=0.95)     # threshold based search
    # the results are returned as a triplet of 1D arrays lims, D, I, 
    # where result for query i is in I[lims[i]:lims[i+1]] (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
    end_time_range_search = time.time()
    print("Time taken to search: %s" % (end_time_range_search - start_time))
    
    posting_list = {}
    inverted_index = {}
    class_label = {}
    for i in range(xb.shape[0]):
        print("result_size for query {0} is {1}".format(i, len(I[limits[i] : limits[i+1]])))
        posting_list[i] = set(I[limits[i] : limits[i+1]])
        class_label[i] = random.randint(1, args.groups)
        for e in posting_list[i]:
            if e not in inverted_index:
                inverted_index[e] = list()
            inverted_index[e].append(i)
    end_time = time.time()
    print("Time taken to generate metadata: %s" % (end_time - start_time))


def create_distance_matrix(X,Y, args):
    D = [[0.0 for _ in range(args.N)] for _ in range(args.N)]
    for i in range(len(D)):
        for j in range(len(D[0])):
            D[i][j] = lp_distance(X[i], Y[j])
            # D[i][j] = spatial.distance.cosine(X[i], Y[j])
    return D

def lp_distance(x1, x2):
    psum = 0.0
    p = 0.5
    for i, j in zip(x1, x2):
        psum += pow(x1 - x2, p)
    
    distance = sum(psum) 
    return pow(distance, 1/p)


def scatter_plot_vectors(xb):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(xb)
    t = reduced.transpose()
    plt.scatter(t[0], t[1])
    plt.show()
    plt.savefig('./scatter_vectors.png')
    plt.cla()
    plt.clf()



if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--N', type=int, required=True)
    # parser.add_argument('--dimension', type=int, required=True)
    # parser.add_argument('--groups', type=int, required=True)
    # args = parser.parse_args()
    # create_dataset(args)
    X = pickle.load(open("/localdisk3/data-selection/cifar-10-vectors", "rb"))
    # faiss_index = FaissSearch(X)
    # faiss_index.threshold_search(0.9)
    scatter_plot_vectors(X)
