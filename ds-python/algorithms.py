"""
Contains code for the greedy fair set cover algorithm 
TODO: change the metadata loading from txt files to on the go faiss search 
"""
import math
import time
import random
import numpy as np
import os
from paths import *
import faiss
from utils_algo import *
import copy

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
    GC = distribution_req # group count tracker
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

def greedyC_random(coverage_factor, distribution_req, q, dataset_name, dataset_size, partition_data, model_name, coveragae_threshold):
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
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    

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
            arr = np.zeros(len(label_ids_to_name.keys()))
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

    end_time = time.time()
    print(len(solution))
    cscore = calculate_cscore(solution, posting_list, delta_size)
    response_time = end_time - start_time
    q.put((solution, cscore, response_time))
    # return solution, posting_list


def greedyNC(coverage_factor, distribution_req, dataset_name, dataset_size, cov_threshold, posting_list):
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
    # location = POSTING_LIST_LOC.format(dataset_name, cov_threshold, 1)
    # posting_list_filepath = location + 'posting_list_alexnet.txt'
    # posting_list_file = open(posting_list_filepath, 'r')
    label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}

    # generate posting list map
    # posting_list = dict()
    # delta = set()
    # lines = posting_list_file.readlines()
    # delta_size = dataset_size
    # for line in lines:
    #     pl = line.split(':')
    #     key = int(pl[0])
    #     value = pl[1].split(',')
    #     value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
    #     arr = np.zeros(delta_size)
    #     arr[value] = 1
    #     posting_list[key] = arr
    #     delta.add(key)
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
            arr = np.zeros(len(label_ids_to_name.keys()))
            arr[label] = 1
            labels_dict[key] = arr
    
    label_file.close()
    CC = np.empty(delta_size) # coverage tracker
    CC[list(delta)] = coverage_factor
    GC = np.array(distribution_req) # group count tracker
    solution = set() # solution set 
    # main loop
    while ((not check_for_zeros(CC)) or (not check_for_zeros(GC))) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # toDo: optimize this loop using scipy
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
    return set(coreset), 0, (start_time-end_time)

def herding(dataset_name, coverage_factor, distribution_req, dataset_size, cov_threshold):
    pass

def k_center(dataset_name, coverage_factor, distribution_req, dataset_size, cov_threshold):
    pass

def bandit_algorithm(coverage_factor, distribution_req, dataset_name, dataset_size, cov_threshold):
    '''
    Computes the randomized bandit fair set cover for the entire dataset
    @params
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list directory
    @returns
        solution of teh randomized bandit fair set cover that satisfies the
        coverage and distribution requirements
        coverage score for this solution
        time taken
    '''
    start_time = time.time()
    label_file = label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    location = POSTING_LIST_LOC.format(dataset_name, cov_threshold, 1)
    posting_list_filepath = location + 'posting_list_alexnet.txt'
    posting_list_file = open(posting_list_filepath, 'r')
    
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

    # Class labels for points
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
    mid_time = time.time()
    print('Time Taken for metadata loading:{0}'.format(mid_time - start_time))
    # Initialize variables to keep track of
    not_satisfied = list(delta) # Points whose cov & dist reqs are not met
    CC = np.full((delta_size), coverage_factor) # Coverage counter
    GC = np.array(distribution_req) # Group requirement counter
    solution = set() # Coreset

    while(len(not_satisfied) > 0):
        actions = delta.difference(solution)
        reward_estimate = {a : {"avg": 0, "stdev": 0, "m2": 0} for a in actions}
        for i in range(0, len(not_satisfied)):
            if (len(actions) <= 1):
                break
            # Keep track of best LCB
            best_LCB = float("-inf")
            # Sample random point
            r = random.sample(not_satisfied)
            r = r[0]
            for a in actions:
                # Calculate score for point r
                r_score = 0
                if (posting_list[a][r] == 1):
                    r_score = CC[r] + distritbution_score(GC, labels_dict[r])
                # Update avg and stdev
                old_estimate = reward_estimate[a]
                new_avg = old_estimate["avg"] + (r_score - old_estimate["avg"]) / (i + 1)
                new_m2 = old_estimate["m2"] + (r_score + old_estimate["avg"]) * (r_score + new_avg)
                new_stdev = math.sqrt(new_m2 / (i + 1))
                reward_estimate[a] = {"avg": new_avg, "stdev": new_stdev, "m2": new_m2}
                # Possibly update best LCB
                new_LCB = new_avg - 2 * new_stdev
                if (new_LCB > best_LCB):
                    best_LCB = new_LCB
            # Remove actions that cannot be optimal
            actions = list(filter(lambda a : reward_estimate[a]["avg"] + 2 * reward_estimate[a]["stdev"] >= best_LCB, actions))

        # Choose best action
        best_action = None
        best_UCB = float('-inf')
        for a in actions:
            a_UCB = reward_estimate[a]["avg"] + 2 * reward_estimate[a]["stdev"]
            if (a_UCB > best_UCB):
                best_action = a
                best_UCB = a_UCB
        if (best_action is None):
            print("cannot find a point")
            break
        
        # If best action is found, update all info
        solution.add(best_action)
        CC = np.subtract(CC, posting_list[best_action])
        GC = np.subtract(GC, labels_dict[best_action])
        not_satisfied = list(filter(lambda p : CC[p] > 0 or distritbution_score(GC, labels_dict[p]) > 0, not_satisfied))
    
    end_time = time.time()
    print(len(solution)) 
    cscore = calculate_cscore(solution, posting_list, delta_size)
    res_time = end_time - start_time
    return solution, cscore, res_time