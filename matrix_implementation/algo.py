from re import L
import time
import random
import numpy as np
import os





def check_for_zeros(CC):
    for e in CC:
        if e > 0:
            return False
    return True


def coverage_score(CC, pl):
    return np.dot(CC.T, pl)

def distritbution_score(GC, gl):
    return np.dot(GC.T, gl)

def algorithm(part_id, coverage_factor, distribution_req, sp):
    # k = coverage_factor
    location = "/localdisk3/data-selection/partitioned_data/" + str(sp) + "/"
    posting_list_filepath = location + 'posting_list_alexnet_' + str(part_id) + '.txt'
    posting_list_file = open(posting_list_filepath, 'r')
    label_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    
    
    

    # generate posting list map
    posting_list = dict()
    delta = set()
    lines = posting_list_file.readlines()
    delta_size = len(lines)
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

    print(len(solution))
    return solution