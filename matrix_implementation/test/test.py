from fileinput import filename
import fnmatch
from os.path import isfile, join
import argparse
from importlib_metadata import distribution
import numpy as np
import os



def get_filename(args):
    return str(args.coverage_factor) + "_" + str(args.sample_weight) + "_" + str(args.distribution_req) + "_" + str(args.composable) + '.txt'

def get_coreset(filename):
    f = open(filename, 'r')
    lines  = f.readlines()
    coreset = [int(l.strip()) for l in lines]
    return set(coreset)


def check_for_zeros(CC):
    incorrect = 0
    for e in CC:
        if e > 0:
            incorrect += 1
    return incorrect

def test_coverage(args, coreset):
    delta_size = 50000
    posting_list = dict()
    delta = set()
    
    location = "/localdisk3/data-selection/partitioned_data/" + str(args.sample_weight) + "/"
    for root, dirnames, filenames in os.walk(location):
        for filename in fnmatch.filter(filenames, '*.txt'):
            posting_list_file = open(os.path.join(root, filename), 'r')    
            lines = posting_list_file.readlines()
            for line in lines:
                pl = line.split(':')
                key = int(pl[0])
                if key in coreset:
                    value = pl[1].split(',')
                    value = [int(v.replace("{", "").replace("}", "").strip()) for v in value]
                    arr = np.zeros(delta_size)
                    arr[value] = 1
                    posting_list[key] = arr
                delta.add(key)
    
    assert(len(delta) == (args.sample_weight * delta_size))
    coreset_CC = np.zeros(delta_size)
    for key, value in posting_list.items():
        coreset_CC = np.subtract(coreset_CC, value)

    coverage_satisfied = check_for_zeros(coreset_CC)
    if coverage_satisfied == 0:
        print("Coverage Test Passed.....")
    else:
        print("Coverage Test Failed.....")

def test_distribution_req(args, coreset):
    label_file = open("/localdisk3/data-selection/class_labels_alexnet.txt", 'r')
    label_ids_to_name = {0 : "airplane", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog", 6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"}
    labels = label_file.readlines()
    dist_req = np.zeros(len(label_ids_to_name.keys()))
    dist_req.fill(args.distribution_req)

    for l in labels:
        txt = l.split(':')
        key = int(txt[0].strip())
        label = int(txt[1].strip())
        if key in coreset:
            arr = np.zeros(len(label_ids_to_name.keys()))
            arr[label] = 1
            dist_req = np.subtract(dist_req, arr)
    
    distribution_satisfied = check_for_zeros(dist_req)
    if distribution_satisfied == 0:
        print("Distribution Req Test Passed.....")
    else:
        print("Distribution Req Test Failed.....")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage_factor', type=int, required=True)
    parser.add_argument('--sample_weight', type=float, required=True)
    parser.add_argument('--distribution_req', type=int, required=True)
    parser.add_argument('--composable', type=int, required=True)
    args = parser.parse_args()


    filename = get_filename(args)
    if not isfile(join("../runs/", filename)):
        print("No File found with the following parameters")
        exit(-1)
    print("Testing for file: {0}".format(filename))
    coreset = get_coreset(join("../runs/", filename))
    test_coverage(args, coreset)
    test_distribution_req(args, coreset)
    print()