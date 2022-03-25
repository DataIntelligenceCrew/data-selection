import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics

data = {}

def main():
    
    f = open("/localdisk3/data-selection/posting_list_alexnet.txt", 'r')
    lines = f.readlines()
    for line in lines:
        key_value = line.split(':')
        key = key_value[0]
        value = key_value[1].split(',')
        value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
        data[key] = len(value)

    # print(len(data))
    # plt.bar(data.keys(), data.values())
    # dict_df = {"Set_ID" : data.keys(), "Posting List Size" : data.values()}
    # df = pd.DataFrame.from_dict(dict_df)
    # sns.set(font_scale=0.75)
    # sns.displot(df, x="Set_ID", edgecolor="black")
    # plt.show()
    # plt.savefig('./dist-alexnet.png')



def group_distribution():
    set_cover_sol = open("./global_set_cover_alexNET.txt", 'r')
    label_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    labels = label_file.readlines()
    lines = set_cover_sol.readlines()
    labels_dict = dict()
    sol_dist = dict()
    set_cover = []
    for l in labels:
        txt = l.split(':')
        labels_dict[txt[0].strip()] = label_ids_to_name[int(txt[1].strip())]

    for line in lines:
        point = line.strip()
        group = labels_dict[point]
        set_cover.append(int(point))
        if group not in sol_dist:
            sol_dist[group] = []
        sol_dist[group].append(point)
    
    sol_dist = dict(sorted(sol_dist.items()))

    numbers = [len(v) for v in sol_dist.values()]


    plt.bar(sol_dist.keys(), numbers)
    plt.ylabel("Number of Points")
    plt.xlabel("Classes")
    plt.title("Distribution of Set Cover Soltion (AlexNET, k = 2, C = 0.9)")
    plt.xticks(rotation=65)
    plt.tight_layout()
    plt.savefig('./solution_group_distribution_iterative')
    plt.cla()
    plt.clf()

    f = open("/localdisk3/data-selection/posting_list_alexnet.txt", 'r')
    lines2 = f.readlines()
    sol_coverage_dist = dict()
    for line2 in lines2:
        key_value = line2.split(':')
        key = int(key_value[0])
        if key in set_cover:
            value = key_value[1].split(',')
            value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
            for v in value:
                if v not in sol_coverage_dist:
                    sol_coverage_dist[v] = 0
                sol_coverage_dist[v] += 1
    print(statistics.mean(sol_coverage_dist.values()))
    print(min(sol_coverage_dist.values()))
    print(max(sol_coverage_dist.values()))
    plt.bar(sol_coverage_dist.keys(), sol_coverage_dist.values())
    plt.yscale('log')
    plt.ylabel("Number of times covered")
    plt.xlabel("Point ID")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.savefig('./coverage_dist_iterative')
    plt.cla()
    plt.clf()



def representation_quality():
    label_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    labels = label_file.readlines()
    labels_dict = dict()    
    for l in labels:
        txt = l.split(':')
        labels_dict[int(txt[0].strip())] = int(txt[1].strip())
    
    f = open("/localdisk3/data-selection/posting_list_alexnet.txt", 'r')
    lines2 = f.readlines()
    pl = dict()
    delta = set()
    for line2 in lines2:
        key_value = line2.split(':')
        key = int(key_value[0])
        value = key_value[1].split(',')
        value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
        pl[key] = len(value)
        delta.add(key)



    posting_list_distribution = dict()
    for point in delta:
        group = labels_dict[point]
        if group not in posting_list_distribution:
            posting_list_distribution[group] = 0
        posting_list_distribution[group] += pl[point]
    

    dict_df = {"Label" : posting_list_distribution.keys(), "Posting List Size" : posting_list_distribution.values()}
    df = pd.DataFrame.from_dict(dict_df)
    sns.displot(data=df, x="Label", kind="kde")
    plt.show()
    plt.savefig('./rep_quality_alexNET')
    plt.cla()
    plt.clf()

    print(statistics.mean(posting_list_distribution.values()))
    print(statistics.stdev(posting_list_distribution.values()))
    plt.bar(posting_list_distribution.keys(), posting_list_distribution.values())
    plt.xticks(rotation="vertical")
    plt.savefig('./rep_quality_alexNET_bar')


if __name__=="__main__":
    # main()
    # group_distribution()
    representation_quality()
