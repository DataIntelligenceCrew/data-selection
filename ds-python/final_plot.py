import statistics
import matplotlib.pyplot as plt
from os.path import join, isfile
import argparse
import os
from paths import *
from lfw_dataset import *
from itertools import islice
import pandas as pd 
import seaborn as sns 
import csv
import math
import plot

# distribution_req = [50, 100, 200, 300, 500, 700, 900]
distribution_req = [200]
def get_output_filename(params, i, algo_type, cf):
    return METRIC_FILE.format(params.dataset, cf, i, algo_type, params.model_type)

def find_best_model(filename):
    model_details = {}
    try:
        with open(filename, 'r') as f:
            n = 4
            for line in f:
                if line.startswith('Model ID:'):
                    # [training_time, number_runs, mean_test_acc, stdev_test_acc]
                    model_details[line.strip()] = list(islice(f, n)) 
        f.close()
        # best_model_id = None
        # best_model_acc_data = [float("-inf"), float("-inf"), float("-inf"), float("-inf")] # [training_time, number_runs, mean_test_acc, stdev_acc]
        # print(len(model_details.keys()))
        test_acc, train_time = 0.0, 0.0
        for _, value in model_details.items():
            test_acc += float(value[2].strip().split(":")[1])
            train_time += float(value[0].strip().split(":")[1])

        mean_test_acc = test_acc / len(model_details) 
        mean_train_time = train_time / len(model_details)
    except FileNotFoundError:
        mean_train_time, mean_test_acc = 0, 0

    return (mean_train_time, mean_test_acc)   

def get_class_wise(filename, params):
    # class_wise_accuracy = [0] * params.num_clases
    details = {}
    f = open(filename, 'r')
    l = 0
    for line in f:
        n = 10
        if line.startswith('Class Wise Test Accuracy'):
            details[l] = list(islice(f, n)) 
            l += 1
    f.close()
    # print(class_wise_accuracy)
    acc = []
    for _, value in details.items():
        class_wise_accuracy = [float(i.split('\t')[1].split(':')[1]) / 100 for i in value]
        total = statistics.mean(class_wise_accuracy)
        acc.append([abs(((len(class_wise_accuracy) * acc) - total) / len(class_wise_accuracy) - 1) for acc in class_wise_accuracy])
    
    mean_acc_disparity = list()
    for a1, a2 in zip(acc[0], acc[1]):
        mean_acc_disparity.append((a1 + a2) / 2)


    return mean_acc_disparity


def generate_csv_ml_data(model_data, params):
    csv_model_acc = {}
    csv_model_train_time = {}
    for key, value in model_data.items():
        acc_y = []
        time_y = []
        for m in value:
            time_y.append(m[0])
            acc_y.append(m[1])
        test_acc_stdev = [str(a) for a in acc_y]
        csv_model_acc[key] = test_acc_stdev
        csv_model_train_time[key] = time_y
    
    print('ML Acc Data for {0}'.format(params.dataset))
    for key, value in csv_model_acc.items():
        print(key + ' : ' + str(value))
    print()
    print('ML Train Time Data for {0}'.format(params.dataset))
    for key, value in csv_model_train_time.items():
        print(key + ' : ' + str(value))


def generate_class_data_csv(data, params):
    for key, value in data.items():
        print(key + ' : ' + str(value))


def class_wise_box_plots(params):
    whole_dataset_metric_file = FULL_DATA_RESULTS.format(params.dataset)
    class_data = { 
        'DC' : get_class_wise(DC_metric_files[0], params),
        'GFKC' : get_class_wise(GFKC_metrics_files[0], params),
        'C-GFKC': get_class_wise(CGFKC_metrics_files[0], params),
        'FKCBandit' : get_class_wise(FKCBandit_metric_files[0], params),
        'KCenters' : get_class_wise(k_centersNC_metric_files[0], params),
        'GFKC_nocov': get_class_wise(GFKC_nocov_metrics_files_fair[0], params),
        'Whole Dataset' : plot.get_class_wise_accuracy(whole_dataset_metric_file, params)
    }
    print(class_data['Whole Dataset'])

    df = pd.DataFrame.from_dict(class_data)
    vals, names, xs = [],[],[]
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted
    plt.boxplot(vals, labels=names)
    palette = ['blue', 'orange', 'red', 'green', 'purple', 'brown']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy Disparity")
    sns.set_style("whitegrid")
    medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
    )
    plt.tight_layout()
    plt.show()
    plt.savefig('./figures/disparity_{0}_{1}.png'.format(params.dataset, distribution_req[0]))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fashion-mnist', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used for generating feature vectors')
    parser.add_argument('--distribution_req', type=int, default=100)
    params = parser.parse_args()
    
    

    GFKC_metrics_files = [get_output_filename(params,i,'greedyNC', params.coverage_factor) for i in distribution_req]
    CGFKC_metrics_files = [get_output_filename(params,i,'greedyC_random', params.coverage_factor) for i in distribution_req]
    GFKC_nocov_metrics_files_fair = [get_output_filename(params,i,'greedyNC', 0) for i in distribution_req]
    FKCBandit_metric_files = [get_output_filename(params, i, 'MAB', params.coverage_factor) for i in distribution_req]
    DC_metric_files = [get_output_filename(params, i, 'dc', params.coverage_factor) for i in distribution_req]
    k_centersNC_metric_files = [get_output_filename(params, i, 'k_centersNC', params.coverage_factor) for i in distribution_req]

    # model_data = { 
    #     'DC' : [find_best_model(f) for f in DC_metric_files],
    #     'GFKC' : [find_best_model(f) for f in GFKC_metrics_files],
    #     'C-GFKC': [find_best_model(f) for f in CGFKC_metrics_files],
    #     'FKCBandit' : [find_best_model(f) for f in FKCBandit_metric_files],
    #     'KCenters' : [find_best_model(f) for f in k_centersNC_metric_files],
    #     'GFKC_nocov': [find_best_model(f) for f in GFKC_nocov_metrics_files_fair],
    # }

    # generate_csv_ml_data(model_data, params)

    # class_data = { 
    #     'DC' : [get_class_wise_accuracy(f, params) for f in DC_metric_files],
    #     'GFKC' : [get_class_wise_accuracy(f, params) for f in GFKC_metrics_files],
    #     'C-GFKC': [get_class_wise_accuracy(f, params) for f in CGFKC_metrics_files],
    #     'FKCBandit' : [get_class_wise_accuracy(f, params) for f in FKCBandit_metric_files],
    #     'KCenters' : [get_class_wise_accuracy(f, params) for f in k_centersNC_metric_files],
    #     'GFKC_nocov': [get_class_wise_accuracy(f, params) for f in GFKC_nocov_metrics_files_fair],
    # }

    # generate_class_data_csv(class_data, params)
    class_wise_box_plots(params)


