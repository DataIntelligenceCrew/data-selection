import statistics
import matplotlib.pyplot as plt
from os.path import join, isfile
import argparse
import os
from paths import *
from itertools import islice
import pandas as pd 
import seaborn as sns 

distribution_req = [50,100,200,300,400,500,600,700,800,900]
# distribution_req = [100, 200, 300, 400, 500]

def get_output_filename(params, i, algo_type):
    return METRIC_FILE.format(params.dataset, params.coverage_factor, i, algo_type, params.model_type)



def get_metrics(filename):
    data = {}
    f = open(filename, "r")
    lines = f.readlines()
    for l in lines:
        if l.startswith("Dataset Size:"):
            txt = l.split(":")
            data["delta"] = int(txt[1].strip())
        if l.startswith("Coreset Size:"):
            txt = l.split(":")
            data["solution_size"] = int(txt[1].strip())
        if l.startswith("Coverage Score:"):
            txt = l.split(":")
            data["cscore"] = float(txt[1].strip())
        if l.startswith("Time Taken:"):
            txt = l.split(":")
            data["response_time"] = float(txt[1].strip())
    f.close()
    
    return data


def find_best_model(filename):
    model_details = {}
    with open(filename, 'r') as f:
        n = 4
        for line in f:
            if line.startswith('Model ID:'):
                # [training_time, number_runs, mean_test_acc, stdev_test_acc]
                model_details[line.strip()] = list(islice(f, n)) 
    f.close()
    best_model_id = None
    best_model_acc_data = [float("-inf"), float("-inf"), float("-inf"), float("-inf")] # [training_time, number_runs, mean_test_acc, stdev_acc]
    # print(len(model_details.keys()))
    for key, value in model_details.items():
        test_acc = float(value[2].strip().split(":")[1])
        if test_acc > best_model_acc_data[0]:
            best_model_id = key
            best_model_acc_data[2] = test_acc
            best_model_acc_data[3] = float(value[3].strip().split(":")[1])
            best_model_acc_data[0] = float(value[0].strip().split(":")[1])
            best_model_acc_data[1] = float(value[1].strip().split(":")[1])

    return [best_model_id, best_model_acc_data]




def analysis(params, model_type):
    # for each partition, plot the distribution
    if params.group:
        location = POSTING_LIST_LOC_GROUP.format(params.dataset, params.coverage_threshold, params.partitions, model_type)
    else:
        location = POSTING_LIST_LOC.format(params.dataset, params.coverage_threshold, params.partitions)
    cf_bounds = []
    for i in range(params.partitions):
        posting_list_file_path = location + 'posting_list_' + str(i) + '.txt'
        posting_list_file = open(posting_list_file_path, 'r')
        posting_list = dict()
        lines = posting_list_file.readlines()
        inverted_index = dict()
        for line in lines:
            pl = line.split(':')
            key = int(pl[0])
            value = pl[1].split(',')
            value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
            for v in value:
                if v not in inverted_index:
                    inverted_index[v] = list()
                inverted_index[v].append(key)
            posting_list[key] = len(value)
        posting_list_file.close()
        # max_s = max(posting_list.values())
        # min_s = min(posting_list.values())
        # mean_s = statistics.mean(posting_list.values())
        # stdev_s = statistics.stdev(posting_list.values())
        # print('Max Size: {0}\tMin Size: {1}'.format(max_s, min_s))
        # print('Mean: {0}\tStdev: {1}'.format(mean_s, stdev_s))
        inverted_index_size = [len(v) for v in inverted_index.values()]
        # print('Coverage Threshold Upper Bound: {0}'.format(min(inverted_index_size)))
        # print('*******************************************\n')
        cf_bounds.append(min(inverted_index_size))
    
    return cf_bounds



def analysis_full_data(params):
    location = POSTING_LIST_LOC.format(params.dataset, params.coverage_threshold, 1)
    posting_list_filepath = location + 'posting_list_alexnet.txt'
    positng_list_file = open(posting_list_filepath, 'r')
    lines = positng_list_file.readlines()
    posting_list = dict()
    inverted_index = dict()
    for line in lines:
        pl = line.split(':')
        key = int(pl[0])
        value = pl[1].split(',')
        value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
        for v in value:
            if v not in inverted_index:
                inverted_index[v] = list()
            inverted_index[v].append(key)
        posting_list[key] = len(value)
    
    print('Statistics for Full Data')
    max_s = max(posting_list.values())
    min_s = min(posting_list.values())
    mean_s = statistics.mean(posting_list.values())
    stdev_s = statistics.stdev(posting_list.values())
    print('Max Size: {0}\tMin Size: {1}'.format(max_s, min_s))
    print('Mean: {0}\tStdev: {1}'.format(mean_s, stdev_s))
    inverted_index_size = [len(v) for v in inverted_index.values()]
    print('Coverage Threshold Upper Bound: {0}'.format(min(inverted_index_size)))



def score_method(algo_data, model_data, params):
    coreset_score = algo_data["solution_size"] / params.dataset_size
    time_score = algo_data["response_time"] / 10000
    # print(model_data)
    acc_score = 1 / model_data[1][2]
    return coreset_score + time_score + acc_score

def plot_score(data, params):
    plot_filename = "./figures/{0}_{1}_{2}.png".format('method_score', params.model_type, params.dataset)
    for key, value in data.items():
        plt.plot(distribution_req, value, 'o--', label=key)
        plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Score")
    plt.xlabel("Distribution Requirement")
    plt.title("Method Score for k={0} dataset={1}(Lower is Better)".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()


def plot_coreset_metrics(data, params):
    for key, value in data.items():
        size_y = [v["solution_size"]/params.dataset_size for v in value]
        plt.plot(distribution_req, size_y, 'o--', label=key)
        plt.legend()

    plot_filename = "./figures/size_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    plt.xticks(distribution_req)
    plt.ylabel("Coreset Size(Percentage of Original Dataset)")
    plt.xlabel("Distribution Requirement")
    plt.title("Coreset Size for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

    plot_filename = "./figures/time_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    for key, value in data.items():
        time_y = [v["response_time"]/60 for v in value]
        plt.plot(distribution_req, time_y, 'o--', label=key)
    plt.xticks(distribution_req)
    plt.ylabel("Time(Minutes)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()



def plot_ml_metrics(data, params):
    for key, value in data.items():
        acc_y = []
        stdev_y = []
        for m in value:
            acc_y.append(m[1][2])
            stdev_y.append(m[1][3])
        
        plt.errorbar(distribution_req, acc_y, yerr=stdev_y, label=key)
        plt.legend()
    
    plot_filename = "./figures/ml_acc_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    plt.xticks(distribution_req)
    plt.ylabel("Accuracy")
    plt.xlabel("Distribution Requirement")
    plt.title("Model Accuracy Convnet for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.85, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet', help='model used for generating feature vectors')
    params = parser.parse_args()
    if params.dataset == 'mnist':
        params.dataset_size = 60000
        params.num_classes = 10
    elif params.dataset == 'cifar10':
        params.num_classes = 10 
        params.dataset_size = 50000 
    elif params.dataset == 'fashionmnist':
        params.num_classes = 10
        params.dataset_size = 60000
    elif params.dataset == 'cifar100':
        params.dataset_size = 50000
        params.num_classes = 100
    
    



    greedyC_random_metrics_files = [get_output_filename(params,i,'greedyC_random') for i in distribution_req]
    greedyNC_metrics_files = [get_output_filename(params,i,'greedyNC') for i in distribution_req]
    greedyC_group_metric_files = [get_output_filename(params, i, 'greedyC_group') for i in distribution_req]
    random_metric_files = [get_output_filename(params, i, 'random') for i in distribution_req]
    bandit_metric_files = [get_output_filename(params, i, 'MAB') for i in distribution_req]



    coreset_data = { 
        'greedyC_random' : [get_metrics(f) for f in greedyC_random_metrics_files],
        'greedyNC': [get_metrics(f) for f in greedyNC_metrics_files],
        'greedyC_group' : [get_metrics(f) for f in greedyC_group_metric_files],
        'random' : [get_metrics(f) for f in random_metric_files],
        'MAB' : [get_metrics(f) for f in bandit_metric_files]
    }

    model_data = { 
        'greedyC_random' : [find_best_model(f) for f in greedyC_random_metrics_files],
        'greedyNC': [find_best_model(f) for f in greedyNC_metrics_files],
        'greedyC_group' : [find_best_model(f) for f in greedyC_group_metric_files],
        'random' : [find_best_model(f) for f in random_metric_files],
        'MAB' : [find_best_model(f) for f in bandit_metric_files]
    }

    score_data = { 
        'greedyC_random' : [score_method(a, m, params) for a,m in zip(coreset_data['greedyC_random'], model_data['greedyC_random'])],
        'greedyNC': [score_method(a, m, params) for a,m in zip(coreset_data['greedyNC'], model_data['greedyNC'])],
        'greedyC_group' : [score_method(a, m, params) for a,m in zip(coreset_data['greedyC_group'], model_data['greedyC_group'])],
        'random' : [score_method(a, m, params) for a,m in zip(coreset_data['random'], model_data['random'])],
        'MAB' : [score_method(a, m, params) for a,m in zip(coreset_data['MAB'], model_data['MAB'])]
    }
    
    plot_coreset_metrics(data=coreset_data, params=params)
    plot_ml_metrics(data=model_data, params=params)
    plot_score(data=score_data, params=params)
    