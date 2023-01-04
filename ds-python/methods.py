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
import faiss
from paths import * 
from utils_algo import * 
from queue import PriorityQueue


'''
Checks for zeros in the 1-dimensional array a
TODO:  need to see using np.any()
'''
def check_for_zeros(a):
    for e in a:
        if e > 0:
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
    
def cgfkc(solution_queue, K, Q, dataset_name, dataset_size, C, model_name, num_classes, partition_data):
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
    solution_queue.put((coreset, response_time))




def cov_points(i, S, posting_list):
    number_points = [x for x in S if i in posting_list[x]]
    return len(number_points)

def universe(L, S, K, posting_list):
    N_l = list()
    for l in L:
        for x in posting_list[l]:
            N_l.append(x)
    affected_points = [x for x in N_l if cov_points(x, S.difference(L), posting_list) <= K]
    return set(affected_points)


def set_cover(U, coverage_factor, candidates, posting_list, dataset_size):
    CC = np.empty(dataset_size)
    CC[list(U)] = coverage_factor
    coreset = set()
    while (not check_for_zeros(CC)) and len(coreset) < len(candidates):
        best_point, max_score = -1, float('-inf')
        for r in candidates.difference(coreset):
            score_point = coverage_score(CC, posting_list[r])
            if score_point > max_score:
                best_point = r
            
            if best_point == -1:
                print("cannot find a point")
                break
            
            coreset.add(best_point)
            CC = np.clip(np.subtract(CC, posting_list[r]), 0, None)
    
    return coreset


def two_phase(posting_list, coverage_coreset, K, dist_req, dataset_size, dataset_name, num_classes):
    
    '''
    coverage_coreset = run the gfkc with dist_req = 0
    write a subroutine for:
        - set cover
        - finding the universe of a point 
    start the swapping 
    '''
    start_time = time.time()
    delta = set(posting_list.keys())
    delta_minus_coreset = delta.difference(coverage_coreset)
    label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    current_group_req = np.zeros(num_classes)
     # class labels for points
    labels = label_file.readlines()
    labels_dict = dict()
    labels_inverted_index = dict() 
    for l in labels:
        txt = l.split(':')
        key = int(txt[0].strip())
        label = int(txt[1].strip())
        if key in coverage_coreset:
            current_group_req[label] += 1
        if key in posting_list:
            labels_dict[key] = label
            if label not in labels_inverted_index:
                labels_inverted_index[label] = list()
            labels_inverted_index[label].append(key)
    
    curr_coverage_coreset_group_dist = np.subtract(dist_req, current_group_req)
    g_extra =  [i for i,v in enumerate(curr_coverage_coreset_group_dist) if v < 0]
    g_extra_points = {i : -v for i, v in enumerate(curr_coverage_coreset_group_dist) if v < 0}
    g_left = [i for i, v in enumerate(curr_coverage_coreset_group_dist) if v > 0]

    L = [l for l in coverage_coreset if labels_dict[l] in g_extra]
    R = [r for r in delta_minus_coreset if labels_dict[r] in g_left]

    L = set(L)
    R = set(R)
    P_l = PriorityQueue()
    u_l_dict = {}
    for l in L:
        s = set()
        s.add(l)
        u_l = universe(s, coverage_coreset, K, posting_list)
        u_l_dict[l] = u_l
        P_l.put((len(u_l), l))
    
    print('Possible number of replacement points: {0}'.format(P_l.qsize()))
    P_l_pruning_tracker = {i : 0 for i in g_extra}
    P_l_pruned = PriorityQueue()
    while not P_l.empty():
        next_cand = P_l.get()
        point_id = next_cand[1]
        label_id = labels_dict[point_id]
        if P_l_pruning_tracker[label_id] <= g_extra_points[label_id]:
            P_l_pruning_tracker[label_id] += 1
            P_l_pruned.put(next_cand)
        

    print('Possible number of replacement points after pruning: {0}'.format(P_l_pruned.qsize()))
    # TODO: remove points from P_l if s_g > l_g
    fairness_not_satisfied = True
    while not P_l_pruned.empty() and fairness_not_satisfied:
        swap_candidate = P_l_pruned.get()
        cand_id = swap_candidate[1]
        r_star = set_cover(u_l_dict[cand_id], 1, R, posting_list, dataset_size)
        R = R.difference(r_star)
        if len(r_star) > 0:
            coverage_coreset = coverage_coreset.union(r_star)
            coverage_coreset.remove(cand_id)

        current_group_req = np.zeros(num_classes)
        for p in coverage_coreset:
            current_group_req[labels_dict[p]] += 1
        curr_coverage_coreset_group_dist = np.subtract(dist_req, current_group_req)
        g_left = [(i,v) for i, v in enumerate(curr_coverage_coreset_group_dist) if v > 0]
        if len(g_left) == 0:
            fairness_not_satisfied = False

    # TODO: for groups that still have points left, add from the posting list 
    current_group_req = np.zeros(num_classes)
    for p in coverage_coreset:
        current_group_req[labels_dict[p]] += 1
    
    curr_coverage_coreset_group_dist = np.subtract(dist_req, current_group_req)
    g_left = [(i,v) for i, v in enumerate(curr_coverage_coreset_group_dist) if v > 0]

    for groups in g_left:
        group_id = groups[0]
        s_g = groups[1]
        possible_candidates = [posting_list[s].append(s) for s in labels_inverted_index[group_id]]
        min_pl_sort = sorted(possible_candidates, key=len)
        for p in min_pl_sort[:s_g]:
            coverage_coreset.add(p[-1])
    
    coreset = coverage_coreset
    end_time = time.time()
    return coreset, end_time - start_time








