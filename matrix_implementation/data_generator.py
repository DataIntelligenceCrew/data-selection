from cProfile import label
import faiss_search 
import os
from collections import defaultdict
import pickle
import random
from os.path import isfile, join
import numpy as np

class MetadataGenerator:
    def __init__(self, filepath, number_partitions, sample_percentage, sample=False):
        self.filepath = filepath
        self.number_of_partitions = number_partitions
        self.sample = sample
        if self.sample:
            # sample from each class
            self.sp = sample_percentage
            self.create_dataset()


    def create_dataset(self):
        self.sampled = []
        class_labels_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
        labels = class_labels_file.readlines()
        self.labels_dict = dict()
        for l in labels:
            txt = l.split(':')
            key = int(txt[1].strip())
            value = int(txt[0].strip())
            if key not in self.labels_dict:
                self.labels_dict[key] = list()

            self.labels_dict[key].append(value)

        for key, value in self.labels_dict.items():
            k = int(len(value) * self.sp)
            self.sampled += random.sample(value, k)
        
        assert(len(self.sampled) == int(50000 * self.sp))




    def generate_metadata(self, threshold):
        self.feature_vectors = pickle.load(open(self.filepath, "rb"))
        if self.sample:
            # feature_vectors is of shape(N,d) so we take all 
            # the indices that are in self.dataset ==> axis = 0
            self.feature_vectors = np.take(self.feature_vectors, self.sampled, 0)
            self.generate_partitions(len(self.sampled))
        else:
            self.generate_partitions(50000)

        # for each partition generate the metadata and write it to the directory
        for key, value in self.partitions.items():
            faiss = faiss_search.FaissSearch(np.take(self.feature_vectors, value, 0))
            faiss.threshold_search(threshold, key, self.sp)



    def generate_partitions(self, delta_size):
        self.partitions = dict()
        for i in range(delta_size):
            part_id = random.randint(0, self.number_of_partitions - 1)
            if part_id not in self.partitions:
                self.partitions[part_id] = list()
            
            self.partitions[part_id].append(i)
    



    






    

