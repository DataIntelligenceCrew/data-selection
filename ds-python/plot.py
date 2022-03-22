import matplotlib.pyplot as plt
from os.path import join, isfile
import argparse
import os
from paths import *

distribution_req = [50,100,200,300,400,500,600]

def get_output_filename(params, i, algo_type):
    return METRIC_FILE.format(params.dataset, params.coverage_factor, i, algo_type)



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
        if l.startswith("Training Time:"):
            txt = l.split(":")
            data["ml_time"] = float(txt[1].strip())
        if l.startswith("Test Accuracy AlexNet Model:"):
            txt = l.split(":")
            data["ml_acc"] = float(txt[1].strip())
    
    return data

def plot_coreset_size(composable_data, greedy_data, params):
    compsable_y = [data["solution_size"]/(params.dataset_size) for data in composable_data]
    greedy_y = [data["solution_size"]/(params.dataset_size) for data in greedy_data]
    plot_filename = "./figures/size_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    
    plt.plot(distribution_req, compsable_y, 'o--', label="greedyC")
    plt.plot(distribution_req, greedy_y, 'o--', label="greedyNC")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Coreset Size(Percentage of Original Dataset)")
    plt.xlabel("Distribution Requirement")
    plt.title("Coreset Size for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

def plot_coreset_time(composable_data, greedy_data, params):
    compsable_y = [data["response_time"] / 60 for data in composable_data]
    greedy_y = [data["response_time"] / 60 for data in greedy_data]
    plot_filename = "./figures/time_" + str(params.coverage_factor) + "_" + str(params.dataset) + ".png"
    plt.plot(distribution_req, compsable_y, 'o--', label="greedyC")
    plt.plot(distribution_req, greedy_y, 'o--', label="greedyNC")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Time (minutes)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken for k={0} dataset={1}".format(params.coverage_factor, params.dataset))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()





def plot_coreset_ml(composable_data, greedy_data, args):
    composable_ml_time_y = [data["ml_time"] for data in composable_data]
    greedy_ml_time_y = [data["ml_time"] for data in greedy_data]
    composable_ml_acc_y = [data["ml_acc"] for data in composable_data]
    greedy_ml_acc_y = [data["ml_acc"] for data in greedy_data]
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




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
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
    

    greedyC_metrics_files = [get_output_filename(params,i,'greedyC') for i in distribution_req]
    greedyNC_metrics_files = [get_output_filename(params,i,'greedyNC') for i in distribution_req]

    greedyC_data = [get_metrics(f) for f in greedyC_metrics_files]
    greedyNC_data = [get_metrics(f) for f in greedyNC_metrics_files]

    plot_coreset_size(greedyC_data, greedyNC_data, params)
    plot_coreset_time(greedyC_data, greedyNC_data, params)
    # plot_coreset_ml(greedyC_data, greedyNC_data, params)
    # plot_coreset_distribution(composable_data, greedy_data)