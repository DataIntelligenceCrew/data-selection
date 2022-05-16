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

# distribution_req = [50,100,200,300,400,500,600,700,800,900]
# distribution_req = [500]
distribution_req = [50, 100, 200, 300, 500, 700, 900]
# distribution_req = [10, 15, 20, 25, 30, 35, 40, 45, 50]

def get_output_filename(params, i, algo_type, cf):
    # if algo_type.startswith('k_centers') or cf == 0 or i == 0:
    #     return METRIC_FILE.format(params.dataset, cf, i, algo_type, params.model_type)
    # else :
    #     return METRIC_FILE2.format(params.dataset, cf, i, algo_type, params.model_type)
    return METRIC_FILE.format(params.dataset, cf, i, algo_type, params.model_type)

def get_sol_filename(params, i, algo_type, cf):
    if algo_type.startswith('k_centers') or cf == 0 or i == 0:
        return SOLUTION_FILENAME.format(params.dataset, cf, i, algo_type, params.model_type)
    else :
        return SOLUTION_FILENAME2.format(params.dataset, cf, i, algo_type, params.model_type)


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
        if test_acc > best_model_acc_data[2]:
            best_model_id = key
            best_model_acc_data[2] = test_acc
            best_model_acc_data[3] = float(value[3].strip().split(":")[1])
            best_model_acc_data[0] = float(value[0].strip().split(":")[1])
            best_model_acc_data[1] = float(value[1].strip().split(":")[1])

    return [best_model_id, best_model_acc_data]



def find_best_model_list(filename):
    model_details = {}
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
    models = list()
    for key, value in model_details.items():
        test_acc = float(value[2].strip().split(":")[1])
        model_id = key
        models.append((model_id, test_acc))

    print("total models found: {0}\n".format(len(models)))
    return sorted(models, key=lambda x: x[1], reverse=True)




def find_nas_model(params):
    filename = NAS_EXP_FILE.format(params.dataset, params.distribution_req)
    model_details = {}
    with open(filename, 'r') as f:
        n = 2
        for line in f:
            if line.startswith('Model ID:'):
                # [training_time, mean_test_acc]
                model_details[line.strip()] = list(islice(f, n)) 
    f.close()
    nas_models = list()
    total_training_time = 0.0
    # print(len(model_details.keys()))
    for key, value in model_details.items():
        test_acc = float(value[1].strip().split(":")[1])
        total_training_time += float(value[0].strip().split(":")[1])

        model_id = key
        nas_models.append((model_id, test_acc))

    print("Total Time Taken for NAS: {0} hours".format(total_training_time/3600))
    return sorted(nas_models, key=lambda x: x[1], reverse=True)

def get_bias_score(filename, df, full_data=False):
    if full_data:
        df_coreset = df
    else:
        coreset_file = open(filename, 'r')
        lines = coreset_file.readlines()
        coreset = [int(l.strip()) for l in lines]
        coreset_file.close()
        df_coreset = df.iloc[coreset]
    
    df_coreset_male = df_coreset.loc[df_coreset['Male'] == 1]
    df_coreset_female = df_coreset.loc[df_coreset['Male'] == 0]
    return len(df_coreset_male)/len(df_coreset_female)

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

def plot_bias_score(data):
    for key, value in data.items():
        if key == 'Original Dataset':
            plt.plot(distribution_req, value, '--', label=key)
        else:
            plt.plot(distribution_req, value, 'o-', label=key)
        plt.legend()
    
    plt.xlabel('Distribution Requirement')
    plt.ylabel('Bias Ratio')
    plt.xticks(distribution_req)
    plt.tight_layout()
    plt.savefig('./figures/bias_lfw.png')
    plt.cla()
    plt.clf()

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

def generate_csv_dict(coreset_data, params):
    csv_data_size = {}
    csv_data_time = {}
    for key, value in coreset_data.items():
        size_y = [v["solution_size"]/params.dataset_size for v in value]
        time_y = [v["response_time"]/60 for v in value]
        csv_data_size[key] = size_y
        csv_data_time[key] = time_y

    print('Coreset Size Data for {0}'.format(params.dataset))
    for key, value in csv_data_size.items():
        print(key + ' : ' + str(value))
    print()
    print('Coreset Time Data for {0}'.format(params.dataset))
    for key, value in csv_data_time.items():
        print(key + ' : ' + str(value))


def generate_csv_ml_data(model_data, params):
    csv_model_acc = {}
    csv_model_train_time = {}
    for key, value in model_data.items():
        acc_y = []
        stdev_y = []
        time_y = []
        for m in value:
            time_y.append(m[1][0])
            acc_y.append(m[1][2])
            stdev_y.append(m[1][3])
        test_acc_stdev = [str(a) for a in acc_y]
        csv_model_acc[key] = test_acc_stdev
        csv_model_train_time[key] = time_y
    
    print('ML Acc Data for {0}'.format(params.dataset))
    for key, value in csv_model_acc.items():
        print(key + ' : ' + str(value))
    print()
    # print('ML Train Time Data for {0}'.format(params.dataset))
    # for key, value in csv_model_train_time.items():
    #     print(key + ' : ' + str(value))

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
        plt.legend()
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
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.clf()
    plt.cla()



def get_class_wise_accuracy(filename, params):
    # class_wise_accuracy = [0] * params.num_clases
    f = open(filename, 'r')
    for line in f:
        n = 10
        if line.startswith('Class Wise Test Accuracy'):
            class_wise_accuracy = list(islice(f, n)) 
            break
    f.close()
    # print(class_wise_accuracy)
    class_wise_accuracy = [float(i.split('\t')[1].split(':')[1]) / 100 for i in class_wise_accuracy]
    # print(class_wise_accuracy)
    total = statistics.mean(class_wise_accuracy)
    accuracy_parity = [abs(((len(class_wise_accuracy) * acc) - total) / len(class_wise_accuracy) - 1) for acc in class_wise_accuracy]
    # return class_wise_accuracy
    # print(accuracy_parity)
    return accuracy_parity


def plot_class_wise_acc(data, params):
    # for each distribution requirement, plot the class wise accuracy for each algo_type
    whole_dataset_metric_file = FULL_DATA_RESULTS.format(params.dataset)
    whole_dataset_class_wise = get_class_wise_accuracy(whole_dataset_metric_file, params)
    temp = [(c, acc) for c,acc in enumerate(whole_dataset_class_wise)]
    whole_dataset_sorted_acc_disparity = sorted(temp, key=lambda x: x[1])
    x_axis = [i for i in range(params.num_classes)]
    print(whole_dataset_sorted_acc_disparity)
    y_whole = [i[1] for i in whole_dataset_sorted_acc_disparity]
    x = [i[0] for i in whole_dataset_sorted_acc_disparity]
    print(x)
    for idx, dr in enumerate(distribution_req):
        fig, ax = plt.subplots()
        for key, value in data.items():
            acc_disparity = value[idx]
            # print(acc_disparity)
            acc_disparity_acc_whole_trend = [acc_disparity[i] for i in x]
            # print(acc_disparity_acc_whole_trend)
            # plt.plot(x, value[idx], 'o--', label=key)
            ax.plot(x_axis, acc_disparity_acc_whole_trend, 'o', label=key)
            ax.legend()

        ax.plot(x_axis, y_whole, 'o', label='Whole Dataset')
        ax.legend()
        # plt.xticks(x)
        ax.set_xticks(x_axis)
        # ax.set_xticklabels([x[j] for j in x_axis])
        ax.set_xticklabels(x)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy Disparity')
        # plt.ylim(-1, 1)
        # plt.title('Class Wise Accuracy Breakdown for Convnet for DR={0}'.format(dr))
        # plt.ylim(bottom=0)
        plot_filename = "./figures/class_wise/ml_acc_disparity_dr_{0}_dataset_{1}".format(dr, params.dataset)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.cla()
        plt.clf()


def plot_tsne():
    from sklearn.manifold import TSNE
    data, attributes = load_from_disk()
    lfw = np.zeros((13143, 3))
    lfw_y = np.zeros(13143)
    for i in range(13143):
        # need to only get particular feature and it's label
        '''
        "Asian": 0,
        "White": 1,
        "Black": 2,
        '''
        lfw[i] = get_row(i, attributes, data)[3:6]
        if lfw[i][0] == 1:
            label = 0
        elif lfw[i][1] == 1:
            label = 1
        else:
            label = 2
        
        lfw_y[i] = label

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(lfw)
    df = pd.DataFrame()
    df["y"] = lfw_y 
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette('hls', 3), data=df)
    plt.show()
    plt.savefig('./figures/tsne_lfw.png')
    plt.cla()
    plt.clf()


# def plot_gender_detection(params):
#     raise NotImplementedError

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fashion-mnist', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used for generating feature vectors')
    parser.add_argument('--distribution_req', type=int, default=100)
    params = parser.parse_args()
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
        params.num_classes = 2
    

    greedyC_random_metrics_files = [get_output_filename(params,i,'greedyC_random', params.coverage_factor) for i in distribution_req]
    greedyNC_metrics_files = [get_output_filename(params,i,'greedyNC', params.coverage_factor) for i in distribution_req]

    # greedyC_random_metrics_files_fair = [get_output_filename(params,i,'greedyC_random', 0) for i in distribution_req]
    greedyNC_metrics_files_fair = [get_output_filename(params,i,'greedyNC', 0) for i in distribution_req]
    # greedyC_group_metric_files = [get_output_filename(params, i, 'greedyC_group') for i in distribution_req]
    # random_metric_files = [get_output_filename(params, i, 'random') for i in distribution_req]
    bandit_metric_files = [get_output_filename(params, i, 'MAB', params.coverage_factor) for i in distribution_req]
    # bandit_metric_files_fair = [get_output_filename(params, i, 'MAB', 0) for i in distribution_req]
    # stochastic_greedyNC_metric_files = [get_output_filename(params, i, 'stochastic_greedyNC') for i in distribution_req]
    # k_centers_group_metric_files = [get_output_filename(params, i, 'k_centers_group', params.coverage_factor) for i in distribution_req]
    k_centersNC_metric_files = [get_output_filename(params, i, 'k_centersNC', params.coverage_factor) for i in distribution_req]


    coreset_data = { 
        'GCR' : [get_metrics(f) for f in greedyC_random_metrics_files],
        # 'GCR_fairness' : [get_metrics(f) for f in greedyC_random_metrics_files_fair],
        'GNC': [get_metrics(f) for f in greedyNC_metrics_files],
        'GNC_fairness': [get_metrics(f) for f in greedyNC_metrics_files_fair],
        # 'greedyC_group' : [get_metrics(f) for f in greedyC_group_metric_files],
        # 'random' : [get_metrics(f) for f in random_metric_files],
        'MAB' : [get_metrics(f) for f in bandit_metric_files],
        # 'MAB_fairness' : [get_metrics(f) for f in bandit_metric_files_fair],
        # 'stochastic_greedyNC' : [get_metrics(f) for f in stochastic_greedyNC_metric_files],
        # 'k_centers_group' : [get_metrics(f) for f in k_centers_group_metric_files],
        'k_centersNC' : [get_metrics(f) for f in k_centersNC_metric_files]
    }

    model_data = { 
        'GCR' : [find_best_model(f) for f in greedyC_random_metrics_files],
        # 'GCR_fairness' : [find_best_model(f) for f in greedyC_random_metrics_files_fair],
        'GNC': [find_best_model(f) for f in greedyNC_metrics_files],
        'GNC_nocov': [find_best_model(f) for f in greedyNC_metrics_files_fair],
        # 'greedyC_group' : [find_best_model(f) for f in greedyC_group_metric_files],
        # 'random' : [find_best_model(f) for f in random_metric_files],
        'MAB' : [find_best_model(f) for f in bandit_metric_files],
        # 'MAB_fairness' : [find_best_model(f) for f in bandit_metric_files_fair],
        # 'stochastic_greedyNC' : [find_best_model(f) for f in stochastic_greedyNC_metric_files],
        # 'k_centers_group' : [find_best_model(f) for f in k_centers_group_metric_files],
        'k_centersNC' : [find_best_model(f) for f in k_centersNC_metric_files]
    }

    # score_data = { 
    #     'greedyC_random' : [score_method(a, m, params) for a,m in zip(coreset_data['greedyC_random'], model_data['greedyC_random'])],
    #     'greedyNC': [score_method(a, m, params) for a,m in zip(coreset_data['greedyNC'], model_data['greedyNC'])],
    #     'greedyC_group' : [score_method(a, m, params) for a,m in zip(coreset_data['greedyC_group'], model_data['greedyC_group'])],
    #     'random' : [score_method(a, m, params) for a,m in zip(coreset_data['random'], model_data['random'])],
    #     'MAB' : [score_method(a, m, params) for a,m in zip(coreset_data['MAB'], model_data['MAB'])],
    #     'stochastic_greedyNC': [score_method(a, m, params) for a,m in zip(coreset_data['stochastic_greedyNC'], model_data['stochastic_greedyNC'])]
    # }
    
    # # algo_type : [[class_wise_acc for dist = i] for i in distritbution_req]
    class_wise_accuracy_data = {
        'GC': [get_class_wise_accuracy(f, params) for f in greedyC_random_metrics_files],
        # 'GCR_fairness': [get_class_wise_accuracy(f, params) for f in greedyC_random_metrics_files_fair],
        # 'greedyC_group' : [get_class_wise_accuracy(f, params) for f in greedyC_group_metric_files],
        'GNC' : [get_class_wise_accuracy(f, params) for f in greedyNC_metrics_files],
        'GNC_nocov' : [get_class_wise_accuracy(f, params) for f in greedyNC_metrics_files_fair],
        # 'random' : [get_class_wise_accuracy(f, params) for f in random_metric_files],
        'MAB' : [get_class_wise_accuracy(f, params) for f in bandit_metric_files],
        # 'MAB_fairness' : [get_class_wise_accuracy(f, params) for f in bandit_metric_files_fair],
        # 'stochastic_greedyNC' : [get_class_wise_accuracy(f, params) for f in stochastic_greedyNC_metric_files],
        # 'KCG' : [get_class_wise_accuracy(f, params) for f in k_centers_group_metric_files],
        'KCenters' : [get_class_wise_accuracy(f, params) for f in k_centersNC_metric_files],
    }

    # location = '/localdisk3/data-selection/data/metadata/{0}/train.csv'.format(params.dataset)
    # df = pd.read_csv(location)

    # biass_ratio_data = {
    #     'Original Dataset' : [get_bias_score("", df, full_data=True)] * len(distribution_req),
    #     'GNC' : [get_bias_score(f, df) for f in greedyNC_metrics_files],
    #     'MAB' : [get_bias_score(f, df) for f in bandit_metric_files],
    #     'KCNC' : [get_bias_score(f, df) for f in k_centersNC_metric_files]    
    # }

    # plot_bias_score(biass_ratio_data)

    # plot_coreset_metrics(data=coreset_data, params=params)
    # plot_ml_metrics(data=model_data, params=params)
    # plot_score(data=score_data, params=params)
    plot_class_wise_acc(data=class_wise_accuracy_data, params=params)
    # plot_tsne()
    generate_csv_dict(coreset_data, params)
    print()
    generate_csv_ml_data(model_data, params)

    # # NAS Experiments
    # nas_sorted = find_nas_model(params)
    # nas_sorted = find_best_model_list('/localdisk3/data-selection/data/runs/metric_files/lfw_30_50_greedyNC_resnet-18_gd.txt')
    # for n in nas_sorted[:10]:
    #     print('Model ID:{0}\tTest Acc: {1}\n'.format(n[0], n[1]))