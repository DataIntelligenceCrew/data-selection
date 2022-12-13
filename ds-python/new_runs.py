from itertools import islice
import os
from os.path import isfile, isdir, join
import argparse 
from paths import *
import matplotlib.pyplot as plt 
import numpy as np

SEEDS = [1234, 9876, 5555, 1111, 2222, 3333, 4444, 6666, 1010, 0000]
LRS = [0.02, 0.01, 0.001]
# LRS = [0.02]
DR = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
# DR = [50]
# DR = [50, 100, 200, 300, 500, 700, 900]
ALGOS = ['greedyNC', 'greddyC_random', 'k_centers_NC']
ALG2 = ['greedyNC', 'greedyC_random', 'k_centersNC', 'MAB']
ML_ALGS = ['greedyNC', 'greedyC_random', 'MAB']
CT = [0.99, 0.95, 0.85, 0.8, 0.75]

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


def plot_hyperparameter_ml(params) :
    '''
        data = {
            DR = {algname : [model acc across various coverage threshold]}
        }
    '''
    data  ={}

    for dr in DR:
        data[dr] = dict()
        for algo_type in ML_ALGS:
            value_list = []
            for ct in CT:
                metric_file_name = METRIC_FILE2.format(params.dataset, params.coverage_factor, dr, algo_type, ct, params.model_type)
                print(metric_file_name)
                value_list.append(get_model_acc(metric_file_name))
            data[dr][algo_type] = value_list

    print(data)

    



def algo_hyperparameter_analysis(params, algo_type):
    
    # data = {[dist_req] : [(coreset_size, coreset_time), (coreset_size, coreset_time), ...]} over various coverage thresholds
    data = {}
    for dr in DR:
        value_list = []
        for coverage_threshold in CT:
            metric_file_name = METRIC_FILE2.format(params.dataset, params.coverage_factor, dr, algo_type, coverage_threshold, params.model_type)
            f = open(metric_file_name, 'r')
            lines = f.readlines()
            f.close()
            coreset_size, coreset_time = 0, 0
            for line in lines:
                if line.startswith('Coreset Size:'):
                    coreset_size = int(line.split(':')[1].strip())
                if line.startswith('Time Taken:'):
                    coreset_time = float(line.split(':')[1].strip())

            value_list.append((coreset_size, coreset_time))

        data[dr] = value_list
        # plot the data
        # N = len(CT)
        # x = np.arange(N)
        x = CT
        size_y = [(v[0]/params.dataset_size) for v in value_list]
        time_y = [(v[1]/60) for v in value_list]

        size_filename = "./figures/hyperparameter/{0}_{1}_{2}_size.png".format(params.dataset, dr, algo_type)
        time_filename = "./figures/hyperparameter/{0}_{1}_{2}_time.png".format(params.dataset, dr, algo_type) 

        plt.plot(x, size_y, 'o-', label='Coreset Size')
        plt.xlabel('Coverage Threshold')
        plt.ylabel('Coreset Size Ratio')
        # plt.xticks(CT)
        plt.title('Coreset Size Ratio for {0}, DR={1}, CF={2}'.format(params.dataset.upper(), dr, params.coverage_factor))
        plt.tight_layout()
        plt.show()
        plt.savefig(size_filename)
        plt.cla()
        plt.clf()

        plt.plot(x, time_y, 'o-', label='Coreset Size')
        plt.xlabel('Coverage Threshold')
        plt.ylabel('Coreset Time (Minutes)')
        # plt.xticks(CT)
        plt.title('Coreset Time for {0}, DR={1}, CF={2}'.format(params.dataset.upper(), dr, params.coverage_factor))
        plt.tight_layout()
        plt.show()
        plt.savefig(time_filename)        
        plt.cla()
        plt.clf()

    return data 


def plot_hyperparameter_analysis(params, data):
    x = CT
    for dr in DR:
        fig, ax = plt.subplots()
        for key, value in data.items():
            coreset_data = value[dr]
            coreset_size_ratio = [(v[0] / params.dataset_size) for v in coreset_data]
            coreset_time = [(v[1] / 60) for v in coreset_data]
            # ax.plot(x, coreset_size_ratio, 'o-', label=key)
            ax.plot(x, coreset_time, 'o-', label=key)
            ax.legend()
        
        plt.xlabel('Coverge Threshold')
        # plt.ylabel('Coreset Size Ratio')
        plt.ylabel('Coreset Time (minutes)')
        # plt.title('Coreset Size Ratio v.s. Coverage Threshold, DR = {0} ({1})'.format(dr, params.dataset))
        plt.title('Coreset Time v.s. Coverage Threshold, DR = {0} ({1})'.format(dr, params.dataset))
        # plt_filename = './figures/hyperparameter/coreset_size_ratio_{1}_{0}.png'.format(dr, params.dataset)
        plt_filename = './figures/hyperparameter/coreset_time_{1}_{0}.png'.format(dr, params.dataset)
        plt.show()
        plt.tight_layout()
        plt.savefig(plt_filename)
        plt.cla()
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    params = parser.parse_args()
    params.coverage_factor = 30
    params.model_type = 'resnet-18'

    if params.dataset == 'mnist':
        params.dataset_size = 60000
        params.num_classes = 10
    elif params.dataset == 'cifar10':
        params.num_classes = 10 
        params.dataset_size = 50000 
    elif params.dataset == 'fashion-mnist':
        params.num_classes = 10
        params.dataset_size = 60000
    elif params.dataset == 'cifar100':
        params.dataset_size = 50000
        params.num_classes = 100
    elif params.dataset == 'lfw':
        params.dataset_size = 13143
        params.num_classes = 72
    elif params.dataset == 'imagenet':
        params.dataset_size = 11060223
        params.num_classes = 10450
    # GFKC_metrics_files = [get_output_filename(params,i,'greedyNC', params.coverage_factor) for i in DR]
    # CGFKC_metrics_files = [get_output_filename(params,i,'greedyC_random', params.coverage_factor) for i in DR]
    # k_centersNC_metric_files = [get_output_filename(params, i, 'k_centersNC', params.coverage_factor) for i in DR]
    # FKCBandit_metric_files = [get_output_filename(params, i, 'MAB', params.coverage_factor) for i in DR]

    # ml_data = {
    #     'GFKC' : [get_model_acc(f) for f in GFKC_metrics_files],
    #     'C-GFKC' : [get_model_acc(f) for f in CGFKC_metrics_files],
    #     # 'KCenters' : [get_model_acc(f) for f in k_centersNC_metric_files]
    #     'FKCBandit' : [get_model_acc(f) for f in FKCBandit_metric_files]
    # }

    # print(ml_data)

    # algo_hyperparameter_analysis(params, 'greedyC_random')

    # algo_hyperparameter_data = {
    #     'GFKC' : algo_hyperparameter_analysis(params, 'greedyNC'),
    #     'C-GFKC' : algo_hyperparameter_analysis(params, 'greedyC_random'),
    #     'FKCBandit' : algo_hyperparameter_analysis(params, 'MAB')
    # }

    # plot_hyperparameter_analysis(params, algo_hyperparameter_data)
    plot_hyperparameter_ml(params)
    