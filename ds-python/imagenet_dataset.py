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


LOC = "/localdisk3/imagenet21k_resized/imagenet21k_train/"

# def pickel_test():
#     fv = np.ones((11060223, MODELS['resnet-18']))
#     f_loc = FEATURE_VECTOR_LOC.format('imagenet', 'resnet-18')
#     f1 = open(f_loc, 'wb')
#     pickle.dump(fv, f1, protocol=4)
#     f1.close()

def get_data_stats():
    subfolders = [f.path for f in os.scandir(LOC) if f.is_dir()]
    # print(len(subfolders))
    img2vec = Img2Vec(cuda=True, model='resnet-18')
    feature_vectors = np.ones((11060223, MODELS['resnet-18']))
    subfolder_index = {}
    labels_dict = np.ones((11060223, 1))
    imagename_idx = {}
    # go over each folder and get all the files
    file_idx = 0
    for idx, subfold in enumerate(subfolders):
        subfolder_index[subfold] = idx
        files = [f for f in listdir(subfold) if isfile(join(subfold, f)) and f.endswith(".JPEG")]
        for f in files:
            try:
                img = Image.open(join(subfold, f))
                vec = img2vec.get_vec(img)
            except RuntimeError:
                img = Image.open(join(subfold, f)).convert(mode="RGB")
                vec = img2vec.get_vec(img)
            img.close()
            feature_vectors[file_idx] = vec
            labels_dict[file_idx] = idx
            imagename_idx[file_idx] = join(subfold, f)
            print(file_idx)
            file_idx += 1
        print("done with fodler number : {0}".format(idx))
    print(feature_vectors.shape)
    f_loc = FEATURE_VECTOR_LOC.format('imagenet', 'resnet-18')
    labels_loc = "/localdisk3/data-selection/data/metadata/imagenet/labels.txt"
    subfolder_indx_loc = "/localdisk3/data-selection/data/metadata/imagenet/subfolder_idx"
    imagename_loc = "/localdisk3/data-selection/data/metadata/imagenet/imagename_idx"
    f1 = open(f_loc, 'wb')
    pickle.dump(feature_vectors, f1, protocol=4)
    f1.close()

    f2 = open(subfolder_indx_loc, 'wb')
    pickle.dump(subfolder_index, f2, protocol=4)
    f2.close()

    f3 = open(labels_loc, 'w')
    for i in range(11060223):
        f3.write('{0} : {1}\n'.format(i, int(labels_dict[i])))
    f3.close()

    f4 = open(imagename_loc, 'wb')
    pickle.dump(imagename_idx, f4, protocol=4)
    f4.close()
    print(file_idx)




if __name__ == '__main__':
    # feature_vector, label_dict = load_data('resnet-18')
    # location = FEATURE_VECTOR_LOC.format('mnist', 'resnet-18')
    # f = open(location, 'wb')
    # pickle.dump(feature_vector, f)
    # f.close()
    # # print(label_dict)
    # location = '/localdisk3/data-selection/data/metadata/mnist/labels.txt'

    # f = open(location, 'w')

    # for i in range(60000):
    #     f.write('{0} : {1}\n'.format(i, int(label_dict[i])))
    
    # f.close()
    get_data_stats()
    # pickel_test()