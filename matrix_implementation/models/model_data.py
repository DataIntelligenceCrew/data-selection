from ast import arg
import enum
import os
from os.path import isfile, isdir, join
import argparse
import shutil

RUNS_DIR = '../runs/'
CIFAR2PNG_LOC = '/localdisk3/data-selection/cifar_png/train/'



def get_coreset_filename(args):
    return str(args.coverage_factor) + "_" + str(args.sample_weight) + "_" + str(args.distribution_req) + "_" + str(args.composable) + '.txt'

def getfilename(id, size=10000):
    # total of 5 batches each with 10,000 images
    # each image can have an index of [0, 9999]
    batch_id = int(id/size)
    index_id = id - (batch_id * size)
    return "data_batch_" + str(batch_id) + "_index_" + str(index_id) + ".png" 


def copy_images(tp, point_files, classes_files, class_names, args):
    # need to fix the directory structure for the model training data
    INPUT_ROOT_DIR = '/localdisk3/data-selection/model-data/sampled_' + str(args.sample_weight) + "/"
    if tp != "full_data":
        INPUT_ROOT_DIR += str(args.distribution_req) + "/" + tp
    else:
        INPUT_ROOT_DIR += tp
    TRAIN_IMG_DIR = INPUT_ROOT_DIR + '/train/'
    os.makedirs(INPUT_ROOT_DIR, exist_ok=True)
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)

    for name in class_names:
        os.makedirs(join(TRAIN_IMG_DIR, name), exist_ok=True)

    
    for f in point_files:
        for key, value in classes_files.items():
            if f in value:
                src_path = join(CIFAR2PNG_LOC, key)
                src_path = join(src_path, f)
                dest_path = join(TRAIN_IMG_DIR, key)
                dest_path = join(dest_path, f)
                shutil.copy(src_path, dest_path)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage_factor', type=int, required=True)
    parser.add_argument('--sample_weight', type=float, required=True)
    parser.add_argument('--composable', type=int, required=True)
    parser.add_argument('--distribution_req', type=int, required=True)
    args = parser.parse_args()


    class_folders_path = [f.path for f in os.scandir(CIFAR2PNG_LOC) if f.is_dir()]
    class_names = [f.name for f in os.scandir(CIFAR2PNG_LOC) if f.is_dir()]
    # print(class_folders_path)
    classes_files = dict()
    for name, path in zip(class_names, class_folders_path):
        onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
        classes_files[name] = onlyfiles
    


    # move the sampled points to the full_data directory
    sp = args.sample_weight
    location = "/localdisk3/data-selection/partitioned_data/" + str(sp) + "/"
    delta = set()
    for part_id in range(10):
        posting_list_filepath = location + 'posting_list_alexnet_' + str(part_id) + '.txt'
        posting_list_file = open(posting_list_filepath, 'r')
        lines = posting_list_file.readlines()
        for l in lines:
            txt = l.split(':')
            key = int(txt[0].strip())
            delta.add(key)
    
    # assert(len(delta) == 5000)

    coreset_file = open(join(RUNS_DIR, get_coreset_filename(args)), 'r')
    lines = coreset_file.readlines()
    coreset = set()
    for l in lines:
        if not l.startswith('Time taken:'):
            point = int(l.strip())
            coreset.add(point)
    # print(len(coreset))
    coreset_point_files = [getfilename(x) for x in coreset]
    delta_point_files = [getfilename(x) for x in delta]
    

    copy_images('full_data', delta_point_files, classes_files, class_names, args)
    copy_images('coreset', coreset_point_files, classes_files, class_names, args)

