import math
import os
import numpy as np
import multiprocessing
import argparse
import faiss
from algorithms import *
from paths import *
from utils_algo import *
import pandas as pd
import tqdm 


def get_labels_nyc():
    df = open('/localdisk3/nyc_1mil_2021-09_updated_labels.csv', 'r')
    lines = df.readlines()
    data = [line.strip() for line in lines]
    df.close()
    del data[0]
    print(len(data))
    labels_dicts = {}
    for row in data:
        # print(row)

        row_s = row.split(',')
        key = int(row_s[0])
        try:
            label = int(row_s[1])
            if label not in labels_dicts.keys():
                labels_dicts[label] = list()
            labels_dicts[label].append(key)
        except ValueError:
            continue
    # print(labels_dicts)
    counts = {k:len(v) for k,v in labels_dicts.items()}
    print(len(labels_dicts.keys()))
    return labels_dicts, counts



def run_algo(params, dr):
    solution_data = []
    if params.algo_type == 'greedyNC':
        posting_list = get_posting_list_nyc_dist()
        s, cscore, res_time = greedyNC(params.coverage_factor, dr, params.dataset, params.dataset_size, params.coverage_threshold, posting_list, params.num_classes)
        solution_data.append((s, cscore, res_time))

    coreset = set()
    cscores = 0
    response_time = 0
    for t in solution_data:
        coreset = coreset.union(t[0])
        cscores += t[1]
        response_time = max(response_time, t[2])
    
    print('Time Taken:{0}'.format(response_time))
    print('Solution Size: {0}'.format(len(coreset)))
    # report metrics
    metric_file_name = METRIC_FILE2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
    with open(metric_file_name, 'w') as f:
        f.write(
            'Dataset Size: {0}\nCoreset Size: {1}\nCoverage Score: {2}\nTime Taken: {3}\n'.format(
                int(params.dataset_size),
                len(coreset),
                cscores,
                response_time
            )
        )
    f.close()

    # log coreset points
    solution_file_name = SOLUTION_FILENAME2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
    with open(solution_file_name, 'w') as f:
        for s in coreset:
            f.write("{0}\n".format(s))
    f.close()



def combine_posting_lists():
    N = 10 
    dir_loc = "/localdisk3/data-selection/data/metadata/nyc_taxicab/10/loc_pl/{0}"
    posting_list = {}
    progress_bar = tqdm.tqdm(total=N, position=0)
    for i in range(N):
        curr_dir = dir_loc.format(i)
        files = [os.path.join(curr_dir, x) for x in os.listdir(curr_dir)]
        # print(files)
        progress_bar_internal = tqdm.tqdm(total=len(files), position=1)
        for f in files:
            f1 = open(f)
            dist_lines = f1.readlines()
            dist_data = [line.strip().replace('{', '').replace('}', '') for line in dist_lines]
            f1.close()
            dist_dict = convert_to_dict(dist_data)
            for key in dist_dict.keys():
                if key not in posting_list:
                    posting_list[key] = set()
                
                posting_list[key] = posting_list[key].union(dist_dict[key])
        progress_bar_internal.update(1)
    progress_bar.update(1)

    with open('/localdisk3/data-selection/data/metadata/nyc_taxicab/sim_PU_1_DO_1.txt', 'w') as f:
        for key, value in posting_list.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    f.close()



def convert_to_dict(data):
    result = {}
    for d in data:
        pl = d.split(':')
        key = int(pl[0])
        value = pl[1].split(',')
        if value != [' set()']:
            value = [int(v.replace("{", "").replace("}", "").replace("'", '').strip()) for v in value]
        else:
            value = []
        result[key] = set(value)
        # print(values[0])
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nyc_taxicab', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=1, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyNC', help='which algorithm to use')
    parser.add_argument('--distribution_req', type=int, default=0, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=10, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used to produce the feature_vector')
    parser.add_argument('--k', type=int, default=0, help='number of centers for k_centersNC')
    params = parser.parse_args()
    
    params.dataset_size = 1000000
    params.num_classes = 7

    # labels_dict, counts = get_labels_nyc()
    # dr = [params.distribution_req] * params.num_classes
    # dr_updated = [counts[i] if counts[i] < dr[i] else dr[i] for i in range(params.num_classes)]
    # print(dr_updated)

    # print(counts)
    # run_algo(params, dr_updated)
    combine_posting_lists()