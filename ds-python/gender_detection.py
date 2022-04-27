'''
Experiments for Geneder Detection (Section 5.2)
Using LFW+A, CelebA dataset generate coreset based on
race as group labels 
Train a neural network for gender classification
Test on Fairface test set. 
'''

import os
import random
import torch
import faiss
import pickle
from os.path import isfile, isdir
import statistics
import numpy as np
from paths import *
import argparse
from PIL import Image
from img2vec_pytorch import Img2Vec
import json
import matplotlib.pyplot as plt
from networks import * 
from lfw_dataset import *
from main_algo import *
import csv
# from torchvision.models import resnet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import pandas as pd
import shutil
from tensorboardX import SummaryWriter

LFW_LABELS = {'Asian' : 0, 
              'White' : 1, 
              'Black' : 2, 
              'Baby' : 3, 
              'Child' : 4, 
              'Youth' : 5, 
              'Middle Aged' : 6,
              'Senior' : 7, 
              'Indian' : 56
            }

def gather_training_images(params):
    loc = SOLUTION_FILENAME.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.model_type)
    coreset_file = open(loc, 'r')
    lines = coreset_file.readlines()
    coreset = [int(l.strip()) for l in lines]
    coreset_file.close()
    location = '/localdisk3/data-selection/data/metadata/{0}/train.csv'.format(params.dataset)
    df = pd.read_csv(location)
    df_coreset = df.iloc[coreset]
    # df_coreset = df
    df_coreset_male = df_coreset.loc[df_coreset['Male'] == 1]
    df_coreset_female = df_coreset.loc[df_coreset['Male'] == 0]
    print('Statistics for {0}'.format(loc))
    print('Total Points in Coreset: {0}'.format(len(df_coreset)))
    print('# of Males in Coreset: {0}'.format(len(df_coreset_male)))
    print('# of Females in Coreset: {0}'.format(len(df_coreset_female)))

    train_dir = INPUT_IMG_DIR_CORESET.format(params.dataset, params.distribution_req, params.coverage_factor, params.algo_type)
    os.makedirs(train_dir, exist_ok=True)
    # copy male images to train_dir/male/
    copy_images_lfw('male/', df_coreset_male, train_dir)
    copy_images_lfw('female/', df_coreset_female, train_dir)
    # copy female images to train_dir/female/

    # male_dest_path = os.path.join(train_dir, 'male/')
    # os.makedirs(male_dest_path, exist_ok=True)
    # for index, row in df_coreset_male.iterrows():
    #     person_name = row['person'].replace(" ", "_")
    #     img_number = str(row['imagenum']).zfill(4)
    #     src_path = '/localdisk3/data-selection/data/datasets/LFW/lfw/{0}/{0}_{1}.jpg'.format(person_name, img_number)
    #     dest_path = join(male_dest_path, '{0}_{1}.jpg'.format(person_name, img_number))
    #     shutil.copy(src_path, dest_path)



def make_fairface_testset(params):
    test_dir = TEST_IMG_DIR.format(params.dataset)
    fairface_dir = '/localdisk3/data-selection/data/datasets/fairface/'
    os.makedirs(test_dir, exist_ok=True)
    male_test_dir = os.path.join(test_dir, 'male/')
    female_test_dir = os.path.join(test_dir, 'female/')
    os.makedirs(male_test_dir, exist_ok=True)
    os.makedirs(female_test_dir, exist_ok=True)
    fairface_val_csv = open(os.path.join(fairface_dir, 'val.csv'), 'r')
    reader = csv.reader(fairface_val_csv)
    next(reader)
    for row in reader:
        loc = row[0]
        gender = row[2] + '/'
        img_num = loc.split('/')[1]
        gender = row[2]
        if gender.lower() == 'male':
            dest_path = os.path.join(male_test_dir, img_num)
        else:
            dest_path = os.path.join(female_test_dir, img_num)
        
        src_path = os.path.join(fairface_dir, loc)
        shutil.copy(src_path, dest_path)



def copy_images_lfw(label, df, train_dir):
    label_dest_path = os.path.join(train_dir, label)
    os.makedirs(label_dest_path, exist_ok=True)
    for index, row in df.iterrows():
        person_name = row['person'].replace(" ", "_")
        img_number = str(row['imagenum']).zfill(4)
        src_path = '/localdisk3/data-selection/data/datasets/LFW/lfw/{0}/{0}_{1}.jpg'.format(person_name, img_number)
        dest_path = join(label_dest_path, '{0}_{1}.jpg'.format(person_name, img_number))
        shutil.copy(src_path, dest_path)

def generate_coreset(params):
    dist_req = [0] * params.num_classes
    for idx in LFW_LABELS.values():
        dist_req[idx] = params.distribution_req
    
    run_algo(params, dr=dist_req)


def make_data_frame(params):
    data, attributes = load_from_disk()
    df = pd.DataFrame.from_dict(data)
    print(df.head())
    location = '/localdisk3/data-selection/data/metadata/{0}/train.csv'.format(params.dataset)  
    df.to_csv(location)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lfw', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyNC', help='which algorithm to use')
    parser.add_argument('--distribution_req', type=int, default=800, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used to produce the feature_vector')

    # params for model training
    parser.add_argument('--num_runs', type=int, default=20, help="number of runs for model testing")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for model training")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model training")
    parser.add_argument('--seed', type=int, default=1234, help="seed to init torch")

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
    elif params.dataset == 'lfw':
        params.dataset_size = 13143
        params.num_classes = 72
    elif params.dataset == 'celebA':
        params.dataset_size = 202599
        params.num_classes = 41 # this needs to be changed to the only account for the labels used by the algo.

  
    # generate_coreset(params)
    # make_data_frame(params)
    gather_training_images(params)
    print('Done for DR : {0}'.format(params.distribution_req))
    # train_model(params)
    # make_fairface_testset(params)




