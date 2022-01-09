import _pickle as cPickle
# pip install img2vec_pytorch
#from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np

# Loads pickle file (CIFAR-10 data) into dictionary
# See CIFAR-10 docs for the specific format they they use
def unpickle(file):
    with open(file, 'rb') as fo:
        mydict = cPickle.load(fo, encoding='latin1')
    return mydict

# Unpickles all 5 training batches into a list of dictionaries
# Assumes that batches are stored in "../cifar" directory
def unpickle_all():
    dir = 'cifar'
    alldicts = []
    for i in range(5):
        alldicts.append(unpickle(str(dir) + "/data_batch_" + str(i + 1)))
    return alldicts

# Given the dictionary returned from unpickle_all(), and the index of image
# (from 0 to 59999), returns a Pillow image
def get_pil_image(alldicts, index):
    print(index)
    cifar_img = alldicts[index // 10000]["data"][index % 10000] # 1 x 3072 ndarray
    np_image = np.ndarray((32, 32, 3), dtype=np.uint8) # 32 x 32 x 3 ndarray
    for i in range(32):
        for j in range(32):
            offset = i * 32 + j
            np_image[i][j][0] = cifar_img[offset]
            np_image[i][j][1] = cifar_img[1024 + offset]
            np_image[i][j][2] = cifar_img[2048 + offset]
    pil_img = Image.fromarray(np_image, mode="RGB")
    return pil_img

# Return an array of all 50,000 PIL images (only the training data)
def get_all_pil_images(alldicts):
    all_images = []
    for i in range(50000):
        all_images.append(get_pil_image(alldicts, i))
    return all_images

#TODO: the above functions for the test batch

# Main function for testing
if __name__ == "__main__":
    alldicts = unpickle_all()
    get_all_pil_images(alldicts)
