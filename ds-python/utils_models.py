'''
This files copies the requires images to the location for model data based on coreset solution
TODO: currently it only works for cifar10, add other datasets
'''
import os
from os.path import isfile, isdir, join
import argparse
import shutil
from paths import *


# TODO: change size according to dataset, currently it's based on cifar10
def getfilename(id, size=10000):
    # total of 5 batches each with 10,000 images
    # each image can have an index of [0, 9999]
    batch_id = int(id/size)
    index_id = id - (batch_id * size)
    return "data_batch_" + str(batch_id) + "_index_" + str(index_id) + ".png" 


def get_mnist_filename(id):
    return "{0}.png".format(id)

def copy_images(point_files, classes_files, class_names, params, full_data_loc):
    # TRAIN_IMG_DIR = INPUT_IMG_DIR_CORESET.format(params.dataset, params.distribution_req, params.coverage_factor, params.algo_type)
    TRAIN_IMG_DIR = INPUT_IMG_DIR_CORESET2.format(params.dataset, params.distribution_req, params.coverage_factor, params.algo_type, params.coverage_threshold)
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)

    for name in class_names:
        os.makedirs(join(TRAIN_IMG_DIR, name), exist_ok=True)

    
    for f in point_files:
        for key, value in classes_files.items():
            if f in value:
                src_path = join(full_data_loc, key)
                src_path = join(src_path, f)
                dest_path = join(TRAIN_IMG_DIR, key)
                dest_path = join(dest_path, f)
                shutil.copy(src_path, dest_path)




if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # data selection parameters
    parser.add_argument('--dataset', type=str, default='fashion-mnist', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='MAB', help='which algorithm to use [greedyNC, greedyC, MAB, random, herding, k_center, forgetting]')
    parser.add_argument('--distribution_req', type=int, default=200, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18')
    params = parser.parse_args()

    full_data_loc = INPUT_IMG_DIR_FULLDATA.format(params.dataset)
    class_folders_path = [f.path for f in os.scandir(full_data_loc) if f.is_dir()]
    class_names = [f.name for f in os.scandir(full_data_loc) if f.is_dir()]
    # print(class_folders_path)
    classes_files = dict()
    for name, path in zip(class_names, class_folders_path):
        onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
        classes_files[name] = onlyfiles
    

    coreset_file = open(SOLUTION_FILENAME2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type), 'r')
    lines = coreset_file.readlines()
    coreset = set()
    for l in lines:
        point = int(l.strip())
        coreset.add(point)


    coreset_point_files = [getfilename(x) for x in coreset]

    copy_images(coreset_point_files, classes_files, class_names, params, full_data_loc)
    print('Done Copying Images for Filename:{0}'.format(SOLUTION_FILENAME2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)))
