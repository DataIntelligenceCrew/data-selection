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
# distribution_req = [50]

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

def plot_coreset_size(greedyC_data, greedyNC_data, greedyC_group_data, params):
    compsable_y = [data["solution_size"]/(params.dataset_size) for data in greedyC_data]
    greedy_y = [data["solution_size"]/(params.dataset_size) for data in greedyNC_data]
    greedy_y_group = [data["solution_size"]/(params.dataset_size) for data in greedyC_group_data]
    plot_filename = "./figures/size_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    
    plt.plot(distribution_req, compsable_y, 'o--', label="greedyC")
    plt.plot(distribution_req, greedy_y, 'o--', label="greedyNC")
    plt.plot(distribution_req, greedy_y_group, 'o--', label='greedyC_group_partitions')
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Coreset Size(Percentage of Original Dataset)")
    plt.xlabel("Distribution Requirement")
    plt.title("Coreset Size for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

def plot_coreset_time(greedyC_data, greedyNC_data, greedyC_group_data, params):
    compsable_y = [data["response_time"] / 60 for data in greedyC_data]
    greedy_y = [data["response_time"] / 60 for data in greedyNC_data]
    greedy_y_group = [data["solution_size"]/(params.dataset_size) for data in greedyC_group_data]
    
    plot_filename = "./figures/time_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    plt.plot(distribution_req, compsable_y, 'o--', label="greedyC")
    plt.plot(distribution_req, greedy_y, 'o--', label="greedyNC")
    plt.plot(distribution_req, greedy_y_group, 'o--', label='greedyC_group_partitions')
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Time (minutes)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()


def plot_coreset_ml(greedyC_data, greedyNC_data, args):
    composable_ml_time_y = [data["ml_time"] for data in greedyC_data]
    greedy_ml_time_y = [data["ml_time"] for data in greedyNC_data]
    composable_ml_acc_y = [data["ml_acc"] for data in greedyC_data]
    greedy_ml_acc_y = [data["ml_acc"] for data in greedyNC_data]
    full_data_time = 0.0
    full_data_acc = 0.0
    full_data_metric_file = '../runs/metrics_full_data_ml_' + str(args.sample_weight) + ".txt"
    with open(full_data_metric_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("Time Taken to Train AlexNet Model:"):
                txt = l.split(":")
                full_data_time = float(txt[1].strip())
                print("here")
            if l.startswith("Test Accuracy AlexNet Model:"):
                txt = l.split(":")
                full_data_acc = float(txt[1].strip())
    
    full_data_time_y = [full_data_time] * len(distribution_req)
    full_data_acc_y = [full_data_acc] * len(distribution_req)

    # Training Time 
    plot_filename = "../plots/figures/ml_time_" + str(args.coverage_factor) + "_" + str(args.sample_weight) + ".png"
    plt.plot(distribution_req, composable_ml_time_y, 'o--', label="Composable")
    plt.plot(distribution_req, greedy_ml_time_y, 'o--', label="Baseline")
    plt.plot(distribution_req, full_data_time_y, 'o--', label="Full Data")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Time (seconds)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken to train AlexNet Model for k={0} sample_weight={1}".format(args.coverage_factor, args.sample_weight))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

    # Model Accuracy 
    plot_filename = "../plots/figures/ml_acc_" + str(args.coverage_factor) + "_" + str(args.sample_weight) + ".png"
    plt.plot(distribution_req, composable_ml_acc_y, 'o--', label="Composable")
    plt.plot(distribution_req, greedy_ml_acc_y, 'o--', label="Baseline")
    plt.plot(distribution_req, full_data_acc_y, 'o--', label="Full Data")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Model Accuracy")
    plt.xlabel("Distribution Requirement")
    plt.title("AlexNet Model Accuracy for k={0} sample_weight={1}".format(args.coverage_factor, args.sample_weight))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

def plot_coreset_distribution(composable_data, greedy_data):
    pass


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
    print(len(model_details.keys()))
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

        # dict_df = {"Points" : posting_list.keys(), "Posting List Size" : posting_list.values()}
        # df = pd.DataFrame.from_dict(dict_df)
        # sns.displot(data=df, x="Points", kind="kde")
        # plt.bar(posting_list.keys(), posting_list.values(), width=1)
        # plt.xticks(rotation=90)
        # plt.ylabel("Posting List Size")
        # plt.show()
        # plot_file_name = './figures/posting_list_alexnet_' + str(i) + '.png'
        # plt.savefig(plot_file_name)
        # plt.cla()
        # plt.clf()
        # print('Statistics for Parition Number: {0}'.format(i))
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


def plot_cf_upper_bound(params,data):
    y_axis = [i for i in range(params.num_classes)]
    fig, ax = plt.subplots()
    for key, value in data.items():
        label = key + '_' + str(MODELS[key])
        ax.plot(y_axis, value, 'o--', label=label)
    
    ax.set_xlabel('Groups')
    ax.set_ylabel('Coverage Factor Upper Bound')
    ax.legend()
    ax.set_title('Analysis for coverage threshold = {0}'.format(params.coverage_threshold))
    plt.savefig('./figures/cf_analysis.png')
    plt.cla()
    plt.clf()    

def plot_for_one(algo_data, model_data, params):
    # plotting algorithm graphs
    y_size = [data["solution_size"]/(params.dataset_size) for data in algo_data]
    y_time = [data["response_time"] for data in algo_data]

    # plot coreset size
    plot_filename = "./figures/{0}_{1}_{2}_{3}.png".format('size', params.algo_type, params.model_type, params.dataset)
    plt.plot(distribution_req, y_size, 'o--', label='greedyC_group')
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Coreset Size(Percentage of Original Dataset)")
    plt.xlabel("Distribution Requirement")
    plt.title("Coreset Size for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

    # plot coreset time
    plot_filename = "./figures/{0}_{1}_{2}_{3}.png".format('time', params.algo_type, params.model_type, params.dataset)
    plt.plot(distribution_req, y_time, 'o--', label='greedyC_group')
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Time Taken(seconds)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()


    # plot ml_acc
    acc_y = []
    stdev_y = []
    time_y = []
    for m in model_data:
        # print(len(m))
        time_y.append(m[1][1])
        acc_y.append(m[1][2])
        stdev_y.append(m[1][3])

    plot_filename = "./figures/{0}_{1}_{2}_{3}.png".format('ml_time', params.algo_type, params.model_type, params.dataset)
    plt.plot(distribution_req, time_y, 'o--')
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Time Taken(seconds)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken to Train Convnet for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()


    plot_filename = "./figures/{0}_{1}_{2}_{3}.png".format('ml_acc', params.algo_type, params.model_type, params.dataset)
    plt.errorbar(distribution_req, acc_y, yerr=stdev_y)
    plt.legend()
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
    
    # params.group = True
    # cf_data = dict()
    # for key, value in MODELS.items():
    #     cf_data[key] = analysis(params, key)
    
    # location = "/localdisk3/data-selection/data/metadata/{0}/{1}/{2}/cf_analysis.txt".format(params.dataset, params.coverage_threshold, params.partitions)
    # with open(location, 'w') as f:
    #     for key, value in cf_data.items():
    #         f.write("Model Name:{0}\tFV_Dimension:{1}\tCF_UpperBound:{2}\n".format(
    #             key,
    #             MODELS[key],
    #             min(value)
    #         ))
    # f.close()
    # plot_cf_upper_bound(params, cf_data)
    # analysis_full_data(params)
    # greedyC_metrics_files = [get_output_filename(params,i,'greedyC') for i in distribution_req]
    greedyNC_metrics_files = [get_output_filename(params,i,'greedyNC') for i in distribution_req]
    greedyC_group_metric_files = [get_output_filename(params, i, 'greedyC_group') for i in distribution_req]
    
    # greedyC_data = [get_metrics(f) for f in greedyC_metrics_files]
    # greedyNC_data = [get_metrics(f) for f in greedyNC_metrics_files]
    greedyC_group_data = [get_metrics(f) for f in greedyC_group_metric_files]
    greedyC_group_models = [find_best_model(f) for f in greedyC_group_metric_files]
    # print(greedyC_group_models)
    params.algo_type = 'greedyC_group'
    plot_for_one(greedyC_group_data, greedyC_group_models, params)
    # plot_coreset_size(greedyC_data, greedyNC_data, greedyC_group_data, params)
    # plot_coreset_time(greedyC_data, greedyNC_data, greedyC_group_data, params)
    # # plot_coreset_ml(greedyC_data, greedyNC_data, params)
    # # plot_coreset_distribution(composable_data, greedy_data)

    # # find best model arch for coreset
    # greedyC_best_models = [find_best_model(f) for f in greedyC_metrics_files]
    # greedyNC_best_models = [find_best_model(f) for f in greedyNC_metrics_files]

    # print(greedyC_best_models)
    # print(greedyNC_best_models)