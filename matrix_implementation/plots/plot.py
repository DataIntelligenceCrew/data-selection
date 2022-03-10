import matplotlib.pyplot as plt
from os.path import join, isfile
import argparse
import os

distribution_req = [5, 10, 20, 40, 80]

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
    
    return data

def plot_coreset_size(composable_data, greedy_data, args):
    compsable_y = [data["solution_size"] for data in composable_data]
    greedy_y = [data["solution_size"] for data in greedy_data]
    plot_filename = "../plots/figures/size_" + str(args.coverage_factor) + "_" + str(args.sample_weight) + ".png"
    
    plt.plot(distribution_req, compsable_y, 'o--', label="Composable")
    plt.plot(distribution_req, greedy_y, 'o--', label="Baseline")
    plt.legend()
    plt.xticks(distribution_req)
    plt.ylabel("Coreset Size")
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

    plot_coreset_size(composable_data, greedy_data, args)
    plot_coreset_time(composable_data, greedy_data, args)
    # plot_coreset_distribution(composable_data, greedy_data)