"""
Contains code for the greedy fair set cover algorithm 
TODO: change the metadata loading from txt files to on the go faiss search 
"""
import math
import time
import random
import numpy as np
import os
from lfw_dataset import get_row, load_from_disk
from paths import *
import faiss
from utils_algo import *
import copy
from lfw_dataset import *

def calculate_cscore(solution, posting_list, delta_size=50000):
    cmatrix = np.zeros(shape=(len(solution), delta_size))
    for idx, s in enumerate(solution):
        cmatrix[idx] = posting_list[s]
    
    new_matrix = np.dot(cmatrix.T, cmatrix)
    return np.trace(new_matrix)

def check_for_zeros(CC):
    for e in CC:
        if e > 0:
            return False
    return True

def coverage_score(CC, pl):
    return np.dot(CC.T, pl)

def distritbution_score(GC, gl):
    return np.dot(GC.T, gl)


def greedyC_group(part_id, coverage_factor, distribution_req, q, dataset_name, partitions, dataset_size, cov_threshold, model_name):
    '''
    Computes the greedy fair set cover for the given partition
    @params
        part_id : partition id
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list
    @returns
        solution of the greedy fair set cover that satisfies the coverage and 
        distribution requirements
        coverage score for this solution
        time taken
    '''
    start_time = time.time()
    location = POSTING_LIST_LOC_GROUP.format(dataset_name, cov_threshold, partitions, model_name)
    posting_list_filepath = location + 'posting_list_' + str(part_id) + '.txt'
    posting_list_file = open(posting_list_filepath, 'r')
    # label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    # label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    
    # generate posting list map
    posting_list = dict()
    delta = set()
    lines = posting_list_file.readlines()
    delta_size = dataset_size
    for line in lines:
        pl = line.split(':')
        key = int(pl[0])
        value = pl[1].split(',')
        value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
        arr = np.zeros(delta_size)
        arr[value] = 1
        posting_list[key] = arr
        delta.add(key)

    # class labels for points
    # labels = label_file.readlines()
    # labels_dict = dict()    
    # for l in labels:
    #     txt = l.split(':')
    #     key = int(txt[0].strip())
    #     label = int(txt[1].strip())
    #     if key in posting_list:
    #         arr = np.zeros(len(label_ids_to_name.keys()))
    #         arr[label] = 1
    #         labels_dict[key] = arr
    
    delta_list = list(delta)
    CC = np.zeros(delta_size) # coverage tracker
    CC[delta_list] = coverage_factor
    GC = np.array(distribution_req) # group count tracker
    solution = set() # solution set 

    # main loop
    while ((not check_for_zeros(CC)) or (GC > 0)) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # TODO: optimize this loop using scipy
        for p in delta.difference(solution):
            p_score = coverage_score(CC, posting_list[p]) + GC
            if p_score > max_score:
                max_score = p_score
                best_point = p

        if best_point == -1:
            print("cannot find a point")
            break

        solution.add(best_point)
        CC = np.subtract(CC, posting_list[best_point])
        GC = GC - 1

    end_time = time.time()
    print(len(solution))
    cscore = calculate_cscore(solution, posting_list, delta_size)
    response_time = end_time - start_time
    q.put((solution, cscore, response_time))

def greedyC_random(coverage_factor, distribution_req, q, dataset_name, dataset_size, partition_data, model_name, coveragae_threshold, num_classes):
    '''
    Computes the greedy fair set cover for the given partition
    @params
        part_id : partition id
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list
    @returns
        solution of the greedy fair set cover that satisfies the coverage and 
        distribution requirements
        coverage score for this solution
        time taken
    '''
    start_time = time.time()
    label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    # label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    

    params = lambda : None
    params.dataset = dataset_name
    params.coverage_threshold = coveragae_threshold
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
    CC[delta_list] = coverage_factor
    GC = np.array(distribution_req) # group count tracker
    solution = set() # solution set 
    # main loop
    while ((not check_for_zeros(CC)) or (not check_for_zeros(GC))) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # TODO: optimize this loop using scipy
        for p in delta.difference(solution):
            p_score = coverage_score(CC, posting_list[p]) + distritbution_score(GC, labels_dict[p])
            if p_score > max_score:
                max_score = p_score
                best_point = p

        if best_point == -1:
            print("cannot find a point")
            break

        solution.add(best_point)
        CC = np.subtract(CC, posting_list[best_point])
        GC = np.subtract(GC, labels_dict[best_point])
        if coverage_factor == 0:
            CC = np.clip(np.subtract(CC, posting_list[best_point]), 0, None)

    end_time = time.time()
    print(len(solution))
    cscore = calculate_cscore(solution, posting_list, delta_size)
    response_time = end_time - start_time
    q.put((solution, cscore, response_time))
    # return solution, posting_list


def greedyNC(coverage_factor, distribution_req, dataset_name, dataset_size, cov_threshold, posting_list, num_classes):
    '''
    Computes the greedy fair set cover for the entire dataset
    @params
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list directory
    @returns
        solution of the greedy fair set cover that satisfies the coverage and 
        distribution requirements
        coverage score for this solution
        time taken 
    '''

    
    '''
    TODO: generate metadata on the fly, use the following to pass params to the utils_algo function
        params = lambda : None 
        params.dataset = 'cifar10'
        params.coverage_threshold = 0.9
    '''
    start_time = time.time()
    delta_size = dataset_size
    print(delta_size)
    delta = set(posting_list.keys())
    for key, value in posting_list.items():
        arr = np.zeros(delta_size)
        arr[list(value)] = 1
        posting_list[key] = arr
    # delta = set()
    # posting_list_loc = "/localdisk3/data-selection/data/metadata/imagenet/posting_list.txt"
    # f = open(posting_list_loc, 'r')
    # lines = f.readlines()
    # for line in lines:
    #     txt = line.split(':')
    #     key = int(txt[0].strip())
    #     print(txt[1])
    #     value = [int(v.strip()) for v in txt[1]]
    #     arr = np.zeros(delta_size)
    #     arr[list(value)] = 1
    #     posting_list[key] = arr
    #     delta.add(key)
    # f.close()

    # class labels for points
    if dataset_name == 'lfw':
        data, attributes = load_from_disk()
        labels_dict = dict()
        for i in range(delta_size):
            labels_dict[i] = np.array(get_row(i, attributes, data)[3:])
    else:
        label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
        # label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
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
    
        label_file.close()

    CC = np.empty(delta_size) # coverage tracker
    CC[list(delta)] = coverage_factor
    GC = np.array(distribution_req) # group count tracker
    solution = set() # solution set 
    # main loop
    iters = 0
    while ((not check_for_zeros(CC)) or (not check_for_zeros(GC))) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # toDo: optimize this loop using scipy
        for p in delta.difference(solution):
            # print(iters)
            p_score = coverage_score(CC, posting_list[p]) + distritbution_score(GC, labels_dict[p])
            if p_score > max_score:
                max_score = p_score
                best_point = p

        if best_point == -1:
            print("cannot find a point")
            break

        solution.add(best_point)
        
        CC = np.subtract(CC, posting_list[best_point])
        GC = np.subtract(GC, labels_dict[best_point])
        if coverage_factor == 0:
            CC = np.clip(np.subtract(CC, posting_list[best_point]), 0, None)
        
        iters += 1
    end_time = time.time()
    print(len(solution)) 
    cscore = calculate_cscore(solution, posting_list, delta_size)
    res_time = end_time - start_time
    return solution, cscore, res_time


def stochastic_greedyNC(coverage_factor, distribution_req, dataset_name, dataset_size, cov_threshold, posting_list):
    '''
    Computes the greedy fair set cover for the entire dataset
    @params
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list directory
    @returns
        solution of the greedy fair set cover that satisfies the coverage and 
        distribution requirements
        coverage score for this solution
        time taken 
    '''

    
    '''
    TODO: handle LFW dataset posting list and label_dict generation
    '''
    start_time = time.time()
   


    delta_size = dataset_size
    print(delta_size)
    delta = set(posting_list.keys())
    for key, value in posting_list.items():
        arr = np.zeros(delta_size)
        arr[list(value)] = 1
        posting_list[key] = arr

    # class labels for points
    if dataset_name == 'lfw':
        data, attributes = load_from_disk()
        labels_dict = dict()
        for i in range(delta_size):
            labels_dict[i] = np.array(get_row(i, attributes, data)[2:])
    else:
        label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
        label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
        labels = label_file.readlines()
        labels_dict = dict()    
        for l in labels:
            txt = l.split(':')
            key = int(txt[0].strip())
            label = int(txt[1].strip())
            if key in posting_list:
                arr = np.zeros(len(label_ids_to_name.keys()))
                arr[label] = 1
                labels_dict[key] = arr
    
        label_file.close()

    CC = np.empty(delta_size) # coverage tracker
    CC[list(delta)] = coverage_factor
    GC = np.array(distribution_req) # group count tracker
    solution = set() # solution set
    stch = 500 # stochastic step size 
    # main loop
    iters = 0
    while ((not check_for_zeros(CC)) or (not check_for_zeros(GC))) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # toDo: optimize this loop using scipy
        stochastic_sample = random.sample(delta.difference(solution), stch)
        # print(iters)
        for p in stochastic_sample:
            p_score = coverage_score(CC, posting_list[p]) + distritbution_score(GC, labels_dict[p])
            if p_score > max_score:
                max_score = p_score
                best_point = p

        if best_point == -1:
            print("cannot find a point")
            break

        solution.add(best_point)
        CC = np.subtract(CC, posting_list[best_point])
        GC = np.subtract(GC, labels_dict[best_point])
        iters += 1

    end_time = time.time()
    print(len(solution)) 
    cscore = calculate_cscore(solution, posting_list, delta_size)
    res_time = end_time - start_time
    return solution, cscore, res_time


def random_algo(labels_dict, distribution_req):
    start_time = time.time()    
    coreset = []
    for key, value in labels_dict.items():
        coreset += random.sample(value, distribution_req)
    end_time = time.time()
    return set(coreset), 0, (end_time-start_time)

def random_algo_lfw(distribution_req):
    data, attributes = load_from_disk()
    coreset = []
    for i, e in enumerate(distribution_req):
        if  e > 0:
            label_points = [idx for idx, en in enumerate(data[attributes[i]]) if en == 1]
            if e > len(label_points):
                coreset += random.sample(label_points, e)
            else:
                coreset += label_points
    return set(coreset), 0, 0



def k_centers_group(coverage_factor, distribution_req, dataset_name, partitions, cov_threshold, model_name, part_id, q):
    # TODO: K-means for each group based on the distribution requirements 

    # generate the NxN distance matrix for each label group 
    # get the points for that group
    start_time = time.time()
    location = POSTING_LIST_LOC_GROUP.format(dataset_name, cov_threshold, partitions, model_name)
    posting_list_filepath = location + 'posting_list_' + str(part_id) + '.txt'
    posting_list_file = open(posting_list_filepath, 'r')    
    delta = list()
    lines = posting_list_file.readlines()
    for line in lines:
        pl = line.split(':')
        key = int(pl[0])
        delta.append(key)    
    
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(dataset_name, model_name), 'rb'))
    fv_partitions = np.take(feature_vectors, delta, 0)

    # get the distance matrix for the fv_partitions
    from scipy.spatial.distance import cdist 
    distance_matrix = cdist(fv_partitions, fv_partitions, metric='euclidean')
    temp_dist = [float("inf")] * len(delta)
    centers = list()

    curr_center = random.randint(0, len(delta) - 1)
    for i in range(distribution_req):
        centers.append(curr_center)
        for j in range(len(delta)):
            # update the distance of the points to their closest centers
            temp_dist[j] = min(temp_dist[j], distance_matrix[curr_center][j])
        # print(centers)
        # get new center point, that maximizes the min distance
        mi = 0
        for k in range(len(delta)):
            if (temp_dist[k] > temp_dist[mi]):
                mi = k

        curr_center = mi
    
    end_time = time.time()
    coreset = [delta[i] for i in centers]
    response_time = end_time - start_time    
    q.put((set(coreset), 0, response_time))


def k_centersNC(k, dataset_name, dataset_size, model_name):
    start_time = time.time()
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format(dataset_name, model_name), 'rb'))
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(feature_vectors, feature_vectors, metric='euclidean')
    temp_dist = [float("inf")] * dataset_size
    centers = list()

    curr_center = random.randint(0, dataset_size - 1)
    for i in range(k):
        centers.append(curr_center)
        for j in range(dataset_size):
            # update the distance of the points to their closest centers
            temp_dist[j] = min(temp_dist[j], distance_matrix[curr_center][j])
        # print(centers)
        # get new center point, that maximizes the min distance
        mi = 0
        for k in range(dataset_size):
            if (temp_dist[k] > temp_dist[mi]):
                mi = k

        curr_center = mi
    
    end_time = time.time()
    coreset = set(centers)
    response_time = end_time - start_time    
    return coreset, 0, response_time


def bandit_algorithm(coverage_factor, distribution_req, dataset_name, 
                     dataset_size, posting_list):
    # These constants can be adjusted
    max_iter = 50
    '''
    Computes the randomized bandit fair set cover for the entire dataset
    @params
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        dataset_name : name of the dataset used
        dataset_size : 
        cov_threshold : 
        model_name : 
    @returns
        solution of the randomized bandit fair set cover that satisfies the
        coverage and distribution requirements,
        coverage score for this solution,
        time taken
    '''
    start_time = time.time()
    delta_size = dataset_size
    print(delta_size)
    delta = set(posting_list.keys())
    for key, value in posting_list.items():
        arr = np.zeros(delta_size)
        arr[list(value)] = 1
        posting_list[key] = arr

    # class labels for points
    if dataset_name == 'lfw':
        data, attributes = load_from_disk()
        labels_dict = dict()
        for i in range(delta_size):
            labels_dict[i] = np.array(get_row(i, attributes, data)[3:])
    else:
        label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
        label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
        labels = label_file.readlines()
        labels_dict = dict()    
        for l in labels:
            txt = l.split(':')
            key = int(txt[0].strip())
            label = int(txt[1].strip())
            if key in posting_list:
                arr = np.zeros(len(label_ids_to_name.keys()))
                arr[label] = 1
                labels_dict[key] = arr
    
        label_file.close()

    not_satisfied = list(delta)
    CC = np.empty(delta_size) # coverage tracker
    CC[list(delta)] = coverage_factor
    GC = np.array(distribution_req) # group count tracker
    solution = set()

    while(len(not_satisfied) > 0):
        actions = list(delta.difference(solution))
        reward_estimate = {a:0.0 for a in actions}
        for i in range(0, max_iter):
            # Sample random point
            r = random.sample(not_satisfied, 1)[0]
            r_score_if_cover = CC[r] + distritbution_score(GC, labels_dict[r])
            # Update reward estimates
            max_estimate = float('-inf')
            for a in actions:
                r_score = posting_list[r][a] and r_score_if_cover
                reward_estimate[a] += r_score
                max_estimate = max(max_estimate, reward_estimate[a])
            actions = list(filter(lambda a : reward_estimate[a] >= max_estimate, actions))
        # Choose best action
        best_action = random.sample(actions, 1)[0]
        # Update all info based on best action
        solution.add(best_action)
        CC = np.clip(np.subtract(CC, posting_list[best_action]), 0, None)
        #print("CC", str(CC))
        GC = np.clip(np.subtract(GC, labels_dict[best_action]), 0, None)
        #print("GC", str(GC))
        not_satisfied = list(filter(lambda p : CC[p] + distritbution_score(GC, labels_dict[p]) > 0, not_satisfied))
        # For testing
        # print(str(len(solution)), "len(a):", str(len(actions)), "len(ns): ", str(len(not_satisfied)))
    
    end_time = time.time()
    print(len(solution))
    cscore = calculate_cscore(solution, posting_list, delta_size)
    res_time = end_time - start_time
    return solution, cscore, res_time
