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

def load_lfw():
    location = '/localdisk3/data-selection/data/datasets/LFW/attr.txt'
    f = open(location, 'r')
    lines = f.readlines()
    attributes = [value.strip() for value in lines[1].split("\t")[1:]]
    # print(len(lines))
    data = {}
    for a in attributes:
        data[a] = list()
    for i in range(2, len(lines)):
        row_data = [value.strip() for value in lines[i].split("\t")]
        for attrib, rdata in zip(attributes, row_data):
            data[attrib].append(rdata)

    # print(len(data[attributes[3]]))
    f.close()
    return data


def print_stats(data):
    attributes = list(data.keys())
    nsize = len(data[attributes[0]])
    print("Dataset Name: Labelled Faces in the Wild, #Images = {0}".format(nsize))
    # for each attribute print it's max, min and stdev
    # from sklearn.preprocessing import MinMaxScaler
    for attrib in attributes[2:]:
        print("------{0}-----".format(attrib))
        try:
            values = [float(v) for v in data[attrib]]
            # print("Max Value:{0}\tMin Value:{1}\tMean Value:{2}\tStdev:{3}\n\n".format(
            # max(values),
            # min(values),
            # statistics.mean(values),
            # statistics.stdev(values) 
            # ))
            mean_value = statistics.mean(values)
            bool_values = list()
            for v in values:
                if v > mean_value:
                    bool_values.append(1)
                else:
                    bool_values.append(0)
            data[attrib] = bool_values
        except ValueError:
            continue
    location = '/localdisk3/data-selection/data/metadata/lfw/attributes.obj'
    f = open(location, 'wb')
    pickle.dump(data, f)
    f.close()


def load_from_disk():
    location = '/localdisk3/data-selection/data/metadata/lfw/attributes.obj'
    f = open(location, 'rb')
    data = pickle.load(f)
    f.close()
    # print(len(data))
    return data, list(data.keys())

def create_config_file(attributes):
    location = './lfw_dr.json'
    attrib_data = {}
    for attrib in attributes[2:]:
        attrib_data[attrib] = 10
    
    f = open(location, 'w')
    json.dump(attrib_data, f)
    f.close()

def get_row(row_id, attributes, data):
    row_data = list()
    for attrib in attributes:
        try:
            row_data.append(data[attrib][row_id])
        except IndexError:
            print(attrib)
    return row_data


def get_fv(row_data, model_name):
    person_name = row_data[0].replace(" ", "_")
    img_number = row_data[1].zfill(4)
    location = '/localdisk3/data-selection/data/datasets/LFW/lfw/{0}/{0}_{1}.jpg'.format(person_name, img_number)
    # print(location)
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    img = Image.open(location)
    # print(type(img))
    img2vec = Img2Vec(cuda=True, model=model_name)
    vec = img2vec.get_vec(img)
    img.close()
    return vec

def create_data_rowwise(attributes, data):
    nsize = 13143
    row_data = dict()
    for i in range(nsize):
        row_data[i] = get_row(i, attributes, data)
    
    location = '/localdisk3/data-selection/data/metadata/lfw/labels.obj'
    f = open(location, 'wb')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__':
    # data = load_lfw()
    # print_stats(data)
    data, attributes = load_from_disk()
    nsize = 13143
    # print(attributes)
    # for key, value in MODELS.items():
    #     print('Generating fvs using {0}'.format(key))
    #     feature_vectors = np.ones((nsize, value))
    #     for i in range(nsize):
    #         row_data = get_row(i, attributes, data)
    #         feature_vectors[i] = get_fv(row_data, key)
        
    #     feature_vectors = np.array(feature_vectors)
    #     location = FEATURE_VECTOR_LOC.format('lfw', key)
    #     f = open(location, 'wb')
    #     pickle.dump(feature_vectors, f)
    #     f.close()
    # create_config_file(attributes)
    create_data_rowwise(attributes[2:], data)
    # print(get_row(0, attributes, data))
        

    # row0 = get_row(0, attributes, data)
    # print(get_fv(row0, 'resnet-18'))

    
    