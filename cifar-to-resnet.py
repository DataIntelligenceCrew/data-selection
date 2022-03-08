import pickle
# pip install img2vec_pytorch
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np


# Loads pickle file (CIFAR-10 data) into dictionary
# See CIFAR-10 docs for the specific format they they use
def unpickle(file):
    with open(file, 'rb') as fo:
        mydict = pickle.load(fo, encoding='bytes')
    return mydict

# Unpickles all 5 training batches into a list of dictionaries
# Assumes that batches are stored in "../cifar" directory
def unpickle_all():
    dir = '/localdisk3/data-selection/cifar'
    alldicts = []
    for i in range(5):
        alldicts.append(unpickle(str(dir) + "/data_batch_" + str(i + 1)))
    return alldicts

# Given the dictionary returned from unpickle_all(), and the index of image
# (from 0 to 59999), returns a Pillow image
def get_pil_image(alldicts, index):
    cifar_img = alldicts[index // 10000][str.encode('data')][index % 10000] # 1 x 3072 ndarray
    np_image = np.ndarray((32, 32, 3), dtype=np.uint8) # 32 x 32 x 3 ndarray
    for i in range(32):
        for j in range(32):
            offset = i * 32 + j
            np_image[i][j][0] = cifar_img[offset]
            np_image[i][j][1] = cifar_img[1024 + offset]
            np_image[i][j][2] = cifar_img[2048 + offset]
    pil_img = Image.fromarray(np_image, mode="RGB")
    return pil_img


def get_pil_image_label(alldicts, index):
    return alldicts[index // 10000][str.encode('labels')][index % 10000]

# Return a list of all 50,000 PIL images (only the training data)
def get_all_pil_images(alldicts):
    all_images = []
    all_labels = []
    for i in range(50000):
        # all_images.append(get_pil_image(alldicts, i))
        all_labels.append(get_pil_image_label(alldicts, i))
    return all_images, all_labels

# TODO: the above functions for the test batch
# TODO: function to get the class_label for each image

# Return a list of ResNet feature vectors given a list of all PIL images
def get_all_resnet(all_images):
    img2vec = Img2Vec(cuda=True, model='alexnet')
    vectors = img2vec.get_vec(all_images)
    # print(vectors.shape)
    return vectors

# Main function for testing
if __name__ == "__main__":
    alldicts = unpickle_all()
    all_images, all_labels = get_all_pil_images(alldicts)
    f = open('/localdisk3/data-selection/class_labels_alexnet.txt', 'w')
    for i,j in enumerate(all_labels):
        f.write(str(i) + ' : ' + str(j) + "\n")
    f.close()
    # batch computing feature vectors
    # batch_size = 32
    # resnet_vectors = np.ones((len(all_images),4096))
    # for i in range(0, len(all_images), batch_size):
    #     resnet_vectors[i : i + batch_size] = get_all_resnet(all_images[i : i + batch_size])
    
    
    # resnet = np.array(resnet_vectors)
    # print(resnet_vectors.shape)
    # f = open('/localdisk3/data-selection/cifar-10-vectors-alexnet', 'wb')
    # pickle.dump(resnet_vectors, f)
    # f.close()
    # benchmark_data_generator.create_dataset
