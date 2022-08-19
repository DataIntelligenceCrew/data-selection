from itertools import islice
import os
from os.path import isfile, isdir, join
import argparse 
from paths import *

SEEDS = [1234, 9876, 5555, 1111, 2222, 3333, 4444, 6666, 1010, 0000]
# LRS = [0.02, 0.01, 0.001]
LRS = [0.02]
# DR = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
# DR = [50]
DR = [50, 100, 200, 300, 500, 700, 900]
ALGOS = ['greedyNC', 'greddyC_random', 'k_centers_NC']

def get_output_filename(params, i, algo_type, cf):
    return METRIC_FILE.format(params.dataset, cf, i, algo_type, params.model_type)

def get_model_id(lr, seed):
    return CONVNET_MODEL_ID.format(256, 4, 'batchnorm', 'leakyrelu', 'avgpooling', 200, lr, 64, seed)

def get_model_acc(filename):
    model_data = {}
    for lr in LRS:
        model_details = {}
        possible_modelids = [get_model_id(lr, seed) for seed in SEEDS]
        # print(possible_modelids)
        # print(filename)
        try:
            with open(filename, 'r') as f:
                n = 17
                for line in f:
                    if line.startswith('Model ID:'):
                        model_id = line.split(':')[1].strip()
                        if model_id in possible_modelids:
                            model_details[model_id] = list(islice(f, n))
            f.close()
            # print(model_details)
            test_acc, train_time = 0.0, 0.0
            for _, value in model_details.items():
                test_acc += float(value[2].strip().split(":")[1])
                train_time += float(value[0].strip().split(":")[1])
            if len(model_details) == 0:
                mean_train_time, mean_test_acc = 0, 0
            else:
                mean_test_acc = test_acc / len(model_details) 
                mean_train_time = train_time / len(model_details)
        except FileNotFoundError:
            mean_train_time, mean_test_acc = 0, 0
        model_data[lr] = (mean_train_time, mean_test_acc)

    return model_data






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    params = parser.parse_args()
    params.coverage_factor = 30
    params.model_type = 'resnet-18'
    GFKC_metrics_files = [get_output_filename(params,i,'greedyNC', params.coverage_factor) for i in DR]
    CGFKC_metrics_files = [get_output_filename(params,i,'greedyC_random', params.coverage_factor) for i in DR]
    k_centersNC_metric_files = [get_output_filename(params, i, 'k_centersNC', params.coverage_factor) for i in DR]

    ml_data = {
        'GFKC' : [get_model_acc(f) for f in GFKC_metrics_files],
        'C-GFKC' : [get_model_acc(f) for f in CGFKC_metrics_files],
        'KCenters' : [get_model_acc(f) for f in k_centersNC_metric_files]
    }

    print(ml_data)
