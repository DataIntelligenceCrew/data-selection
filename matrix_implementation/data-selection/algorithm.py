"""
Contains code for the greedy fair set cover algorithm 
"""
import fnmatch
import time
import random
import numpy as np
import os


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

def composable_algorithm(part_id, coverage_factor, distribution_req, sp, q):
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
    location = "/localdisk3/data-selection/partitioned_data/" + str(sp) + "/"
    posting_list_filepath = location + 'posting_list_alexnet_' + str(part_id) + '.txt'
    posting_list_file = open(posting_list_filepath, 'r')
    label_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    
    # generate posting list map
    posting_list = dict()
    delta = set()
    lines = posting_list_file.readlines()
    delta_size = 50000
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
    while (not check_for_zeros(CC)) and len(solution) < delta_size and (not check_for_zeros(GC)):
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
    response_time = end_time - start_time
    q.put((solution, cscore, response_time))
    # return solution, posting_list



def algorithm(coverage_factor, distribution_req, sp):
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
    start_time = time.time()
    label_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}

    # generate posting list map
    delta_size = 50000
    posting_list = dict()
    delta = set()
    location = "/localdisk3/data-selection/partitioned_data/" + str(sp) + "/"
    for root, dirnames, filenames in os.walk(location):
        for filename in fnmatch.filter(filenames, '*.txt'):
            posting_list_file = open(os.path.join(root, filename), 'r')    
            lines = posting_list_file.readlines()
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
    
    CC = np.empty(delta_size) # coverage tracker
    CC.fill(coverage_factor)
    GC = np.array(distribution_req) # group count tracker
    solution = set() # solution set 
    # main loop
    while (not check_for_zeros(CC)) and len(solution) < delta_size and (not check_for_zeros(GC)):
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