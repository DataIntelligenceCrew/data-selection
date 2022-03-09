from ast import arg
import enum
import os
from os.path import isfile, isdir, join
import argparse
import shutil

RUNS_DIR = '../runs/'
CIFAR2PNG_LOC = '/localdisk3/data-selection/cifar_png/train/'
INPUT_ROOT_DIR = '/localdisk3/data-selection/model-data/sampled_0.1/full_data/'
TRAIN_IMG_DIR = INPUT_ROOT_DIR + 'train/'

def getfilename(id, size=10000):
    # total of 5 batches each with 10,000 images
    # each image can have an index of [0, 9999]
    batch_id = int(id/size)
    index_id = id - (batch_id * size)
    return "data_batch_" + str(batch_id) + "_index_" + str(index_id) + ".png" 




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coreset_solution', type=str, required=True)
    args = parser.parse_args()


    class_folders_path = [f.path for f in os.scandir(CIFAR2PNG_LOC) if f.is_dir()]
    class_names = [f.name for f in os.scandir(CIFAR2PNG_LOC) if f.is_dir()]
    # print(class_folders_path)
    classes_files = dict()
    for name, path in zip(class_names, class_folders_path):
        onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
        classes_files[name] = onlyfiles
    


    # move the sampled points to the full_data directory
    sp = 0.1
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
    
    assert(len(delta) == 5000)

    coreset_file = open(join(RUNS_DIR, args.coreset_solution), 'r')
    lines = coreset_file.readlines()
    coreset = set()
    for l in lines:
        if not l.startswith('Time taken:'):
            point = int(l.strip())
            coreset.add(point)
    # print(len(coreset))
    coreset_point_files = [getfilename(x) for x in coreset]
    delta_point_files = [getfilename(x) for x in delta]
    
    os.makedirs(INPUT_ROOT_DIR, exist_ok=True)
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)

    for name in class_names:
        os.makedirs(join(TRAIN_IMG_DIR, name), exist_ok=True)

    
    for f in delta_point_files:
        for key, value in classes_files.items():
            # print(f)
            if f in value:
                print('here')
                src_path = join(CIFAR2PNG_LOC, key)
                src_path = join(src_path, f)
                dest_path = join(TRAIN_IMG_DIR, key)
                dest_path = join(dest_path, f)
                shutil.copy(src_path, dest_path)

