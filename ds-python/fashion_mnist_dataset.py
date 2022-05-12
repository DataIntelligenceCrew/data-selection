from os import listdir
import pickle
from multiprocessing.sharedctypes import Value
import os
from os.path import isfile, isdir, join
import statistics
import numpy as np
from paths import *
import argparse
import sklearn
from PIL import Image
from img2vec_pytorch import Img2Vec
import json
import matplotlib.pyplot as plt
import numpy as np
import glob


def load_data(model_name):
    location = '/localdisk3/data-selection/data/datasets/fashion-mnist/train/{0}/'
    img2vec = Img2Vec(cuda=True, model=model_name)
    feature_vectors = np.ones((60000, MODELS[model_name]))
    labels_dist = np.ones((60000, 1))
    for i in range(10):
        dir_location = location.format(i)
        files = [f for f in listdir(dir_location) if isfile(join(dir_location, f))]
        # print(files)
        # break;
        for f in files:
            img = Image.open(join(dir_location, f)).convert(mode="RGB")
            vec = img2vec.get_vec(img)
            img.close()
            key = int(f.split('.')[0])
            feature_vectors[key] = vec
            labels_dist[key] = i

    return feature_vectors, labels_dist




if __name__ == '__main__':
    feature_vector, label_dict = load_data('resnet-18')
    location = FEATURE_VECTOR_LOC.format('fashion-mnist', 'resnet-18')
    f = open(location, 'wb')
    pickle.dump(feature_vector, f)
    f.close()
    # print(label_dict)
    location = '/localdisk3/data-selection/data/metadata/fashion-mnist/labels.txt'

    f = open(location, 'w')

    for i in range(60000):
        f.write('{0} : {1}\n'.format(i, int(label_dict[i])))
    
    f.close()