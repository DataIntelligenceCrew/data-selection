'''
Methods to generate FKC Coresets
- Greedy FKC : gfkc()
- Composable Greedy FKC : cgfkc()
- Distributed Submodular Cover : 
- Two Phase Algorithm : 
'''


import time 
import random 
import numpy as np
from paths import * 
from utils_algo import * 



'''
Checks for zeros in the 1-dimensional array a
TODO:  need to see using np.any()
'''
def check_for_zeros(a):
    for e in a:
        if a > 0:
            return False
    return True

'''
Given the current coverage tracker and a posting list,
computes the coverage score for the point
'''
def coverage_score(coverage_tracker, posting_list):
    return np.dot(coverage_tracker.T, posting_list)

'''
Given the current fairness tracker and a group list,
computes the fairness score for the point
'''
def fairness_score(fairness_tracker, group_list):
    return np.dot(fairness_tracker.T, group_list)



def gfkc(K, Q, dataset_name, dataset_size, posting_lists, num_classes):
    start_time = time.time()
    delta_size = dataset_size
    print('Starting FKC Construction using GFKC for {0}, with size : {1}'.format(dataset_name, delta_size))
    # generate the dataset and convert the posting list to binary vectors
    delta = set(posting_lists.keys())
    for point, pl in posting_lists.items():
        arr = np.zeros(delta_size)
        arr[list(pl)] = 1
        posting_lists[point] = arr
    
    # generate the fairness trackers for each point
    if dataset_name == 'lfw':
        pass
    else:
        label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
        labels = label_file.readlines()
        labels_dict = dict()
        for l in labels:
            txt = l.split(':')
            key = int(txt[0].strip())
            label = int(txt[1].strip())
            if key in posting_lists:
                arr = np.zeros(num_classes)
                arr[label] = 1
                labels_dict[key] = arr
        
        label_file.close()

    
    CC = np.empty(delta_size) 
    CC[list(delta)] = K
    GC = np.array(Q)
    coreset = set()

    iters = 0

    while ((not check_for_zeros(CC)) or (not check_for_zeros(GC))) and len(coreset) < delta_size:
        best_point, max_score = -1, float('-inf')
        for p in delta.difference(coreset):
            score_point = coverage_score(CC, posting_lists[p]) + fairness_score(GC, labels_dict[p])
            if score_point > max_score:
                best_point = p
            
            if best_point == -1:
                print("cannot find a point")
                break
            
            coreset.add(best_point)
            CC = np.clip(np.subtract(CC, posting_lists[p]), 0, None)
            GC = np.clip(np.subtract(GC, labels_dict[p]), 0, None)

            iters += 1
    
    end_time = time.time()

    print('Size of the coreset : {0}'.format(len(coreset)))

    response_time = end_time - start_time

    return coreset, response_time
    
def cgfkc(part_id, solution_queue, K, Q, dataset_name, partitions, dataset_size, C, model_name, num_classes, partition_data):
    start_time = time.time()
    label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    params = lambda : None
    params.dataset = dataset_name
    params.coverage_threshold = C
    posting_list = get_posting_lists(params, partition_data, model_name)
    delta_size = dataset_size
    delta = set(posting_list.keys())
    for key, value in posting_list.items():
        arr = np.zeros(delta_size)
        arr[list(value)] = 1
        posting_list[key] = arr

    # class labels for points
    labels = label_file.readlines()
    labels_dict = dict()    
    for l in labels:
        txt = l.split(':')
        key = int(txt[0].strip())
        label = int(txt[1].strip())
        if key in posting_list:
            arr = np.zeros(num_classes)
            arr[label] = 1
            labels_dict[key] = arr
    
    delta_list = list(delta)
    CC = np.zeros(delta_size) # coverage tracker
    CC[delta_list] = K
    GC = np.array(Q) # group count tracker
    coreset = set() # solution set 
    # main loop
    while ((not check_for_zeros(CC)) or (not check_for_zeros(GC))) and len(coreset) < len(delta):
        best_point, max_score = -1, float('-inf')
        # TODO: optimize this loop using scipy
        for p in delta.difference(coreset):
            p_score = coverage_score(CC, posting_list[p]) + fairness_score(GC, labels_dict[p])
            if p_score > max_score:
                max_score = p_score
                best_point = p

        if best_point == -1:
            print("cannot find a point")
            break

        coreset.add(best_point)
        CC = np.clip(np.subtract(CC, posting_list[p]), 0, None)
        GC = np.clip(np.subtract(GC, labels_dict[p]), 0, None)

    end_time = time.time()
    print(len(coreset))
    response_time = end_time - start_time
    q.put((coreset, response_time))