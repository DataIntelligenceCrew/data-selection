import matplotlib.pyplot as plt
from os.path import join, isfile
import argparse
import os

distribution_req = [100,200,300,400,500,600,700,800,900,1000]

def get_output_filename(cf, sp, dist, c):
    return "../runs/metrics_" + str(cf) + "_" + str(sp) + "_" + str(dist) + "_" + str(c) + ".txt"



def get_metrics(filename):
    data = {}
    f = open(filename, "r")
    lines = f.readlines()
    for l in lines:
        if l.startswith("Delta Size:"):
            txt = l.split(":")
            data["delta"] = int(txt[1].strip())
        if l.startswith("Solution Size:"):
            txt = l.split(":")
            data["solution_size"] = int(txt[1].strip())
        if l.startswith("Coverage Score:"):
            txt = l.split(":")
            data["cscore"] = float(txt[1].strip())
        if l.startswith("Time Taken:"):
            txt = l.split(":")
            data["response_time"] = float(txt[1].strip())
        if l.startswith("Time Taken to Train AlexNet Model:"):
            txt = l.split(":")
            data["ml_time"] = float(txt[1].strip())
        if l.startswith("Test Accuracy AlexNet Model:"):
            txt = l.split(":")
            data["ml_acc"] = float(txt[1].strip())
    
    return data

def plot_coreset_size(composable_data, greedy_data, args):
    compsable_y = [data["solution_size"]/(50000 * args.sample_weight) for data in composable_data]
    greedy_y = [data["solution_size"]/(50000 * args.sample_weight) for data in greedy_data]
    plot_filename = "../plots/figures/size_" + str(args.coverage_factor) + "_" + str(args.sample_weight) + ".png"
    
    plt.plot(distribution_req, compsable_y, 'o--', label="Composable")
    plt.plot(distribution_req, greedy_y, 'o--', label="Baseline")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Coreset Size(Percentage of Original Dataset)")
    plt.xlabel("Distribution Requirement")
    plt.title("Coreset Size for k={0} sample_weight={1}".format(args.coverage_factor, args.sample_weight))
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()

def plot_coreset_time(composable_data, greedy_data, args):
    compsable_y = [data["response_time"] for data in composable_data]
    greedy_y = [data["response_time"] for data in greedy_data]
    plot_filename = "../plots/figures/time_" + str(args.coverage_factor) + "_" + str(args.sample_weight) + ".png"
    plt.plot(distribution_req, compsable_y, 'o--', label="Composable")
    plt.plot(distribution_req, greedy_y, 'o--', label="Baseline")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Time (seconds)")
    plt.xlabel("Distribution Requirement")
    plt.title("Time Taken for k={0} sample_weight={1}".format(args.coverage_factor, args.sample_weight))
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
    parser.add_argument('--coverage_factor', type=int, required=True)
    parser.add_argument('--sample_weight', type=float, required=True)
    args = parser.parse_args()

    

    composable_metrics_files = [get_output_filename(args.coverage_factor, args.sample_weight, i, 0) for i in distribution_req]
    greedy_metrics_files = [get_output_filename(args.coverage_factor, args.sample_weight, i, 1) for i in distribution_req]

    composable_data = [get_metrics(f) for f in composable_metrics_files]
    greedy_data = [get_metrics(f) for f in greedy_metrics_files]

    # plot_coreset_size(composable_data, greedy_data, args)
    # plot_coreset_time(composable_data, greedy_data, args)
    plot_coreset_ml(composable_data, greedy_data, args)
    # plot_coreset_distribution(composable_data, greedy_data)