import pickle
from multiprocessing.sharedctypes import Value
import os
from os.path import isfile, isdir
import statistics
import numpy as np
from paths import *
import argparse
import sklearn
from PIL import Image
from img2vec_pytorch import Img2Vec
import json
import matplotlib.pyplot as plt





def load_mat_file(test=False):
    location = '/localdisk3/data-selection/data/datasets/SVHN/{0}/digitStruct.mat'

    if test:
        location = location.format('test')
    else:
        location = location.format('train')
    
    from mat4py import loadmat

    data = loadmat(location)
    return data



















if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    train_mat_file = load_mat_file()
    print(train_mat_file)
    
