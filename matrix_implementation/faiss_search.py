import faiss
import numpy as np
import random
import os

class FaissSearch:
    def __init__(self, feature_vectors):
        self.d = feature_vectors.shape[1]
        self.N = feature_vectors.shape[0]
        self.posting_list = {}
        self.inverted_index = {}
        # self.class_labels = {}
        self.xb = feature_vectors.astype('float32')
        self.xb[:, 0] += np.arange(self.N) / 1000
        self.cpu_index = faiss.IndexFlatL2(self.d)
        faiss.normalize_L2(x=self.xb)
        self.cpu_index.add(self.xb)
        print("successfully build faiss index")
        print("index size: ", self.cpu_index.ntotal)


    def threshold_search(self, threshold, part_id, sp):
        # batch_size = 1000
        # for i in range(0, self.xb.shape[0], batch_size):
        #     limits, D, I = self.cpu_index.range_search(self.xb[i:i+batch_size], threshold)
        #     for j in range(batch_size):
        #         self.posting_list[i+j] = set(I[limits[j] : limits[j+1]])
        
        limits, D, I = self.cpu_index.range_search(self.xb, threshold)
        for i in range(self.xb.shape[0]):
            self.posting_list[i] = set(I[limits[i] : limits[i+1]])
            # for e in self.posting_list[i]:
            #     if e not in self.inverted_index:
            #         self.inverted_index[e] = list()
            #     self.inverted_index[e].append(i)
        
        print("metadata generated")
        location = "/localdisk3/data-selection/partitioned_data/" + str(sp) + "/"
        if not os.path.isdir(location):
            os.makedirs(location)
        posting_list_file = location + 'posting_list_alexnet_' + str(part_id) + '.txt'
        # inverted_index_file = 'inverted_index_alexnet_' + str(part_id) + '.txt'
        self.write_to_file(self.posting_list, posting_list_file)
        # self.write_to_file(self.inverted_index, inverted_index_file)


    def write_to_file(self, d, filepath):
        with open(filepath, 'w') as f:
            for key, value in d.items():
                f.write(str(key) + " : " + str(value) + "\n")
        f.close()
