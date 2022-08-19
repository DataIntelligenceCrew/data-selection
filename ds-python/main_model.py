
from cgi import test
import os
from os.path import isfile
import statistics
import networks
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
import time
from paths import *
import numpy as np
from tensorboardX import SummaryWriter



def get_model_dump_paths(params):
    output_path = MODEL_OUTPUT_DIR.format(params.dataset, params.distribution_req, params.coverage_factor, params.algo_type, params.model)
    tblog_path = output_path + 'tblogs/'
    checkpoint_path = output_path + 'checkpoints/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tblog_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    return tblog_path, checkpoint_path

def get_dataloader(params, test=False):
    if params.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    elif params.dataset == 'mnist':
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    elif params.dataset == 'fashion-mnist':
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        
    elif params.dataset == 'lfw':
        transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       transforms.ToTensor()])
    loc = None
    if test:
        loc = TEST_IMG_DIR.format(params.dataset)
    else:
        if params.coreset == 1:
            loc = INPUT_IMG_DIR_CORESET.format(params.dataset, params.distribution_req, params.coverage_factor, params.algo_type)
        elif params.coreset == 0:
            loc = INPUT_IMG_DIR_FULLDATA.format(params.dataset)
    print(loc)
    dataset = datasets.ImageFolder(loc, transform=transform)
    print(dataset.class_to_idx)
    return data.DataLoader(dataset, shuffle=True, pin_memory=True, num_workers=8, drop_last=True, batch_size=params.batch_size)


class TensorDataset(data.Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def get_dc_dataloader(params):
    if params.dataset == 'fashion-mnist':
        dataset_name = 'FashionMNIST'
    else:
        dataset_name = params.dataset.upper()

    location = os.path.join("/localdisk3/data-selection/dataset-condensation/result/", 'res_%s_%s_%s_%dipc.pt'%('DC', dataset_name, 'ConvNet', params.distribution_req))

    results = torch.load(location)
    data_save = results['data']
    image_syn_train, label_syn_train = data_save[0][0], data_save[0][1]
    dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
    trainloader = data.DataLoader(dst_syn_train, batch_size=params.batch_size, shuffle=True, num_workers=8)

    return trainloader

def get_model_metric_file(params):
    if params.coreset == 1:
        return METRIC_FILE.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.model_type)
    else:
        return FULL_DATA_RESULTS.format(params.dataset)

def get_model_id(params):
    if params.model == 'alexnet':
        return ALEXNET_MODEL_ID.format(params.num_epochs, params.lr, params.batch_size, params.seed)
    elif params.model == 'convnet':
        return CONVNET_MODEL_ID.format(params.net_width, params.net_depth, params.net_norm, params.net_act, params.net_pooling, params.num_epochs, params.lr, params.batch_size, params.seed)
    elif params.model == 'resnet34':
        return RESNET34_MODEL_ID.format(params.num_epochs, params.lr, params.batch_size, params.seed)

def train_and_test(model, params):
    tblog_path, checkpoint_dir = get_model_dump_paths(params)
    model_id = get_model_id(params)
    model_path = os.path.join(checkpoint_dir, '{0}.pt'.format(model_id))
    torch.manual_seed(params.seed)
    print('used seed : {}'.format(params.seed))

    tbwriter = SummaryWriter(log_dir=tblog_path, comment=model_id)
    print('tensorbaordX summary writer created')

    print('model initialized')
    print(model)
    
    if params.algo_type == 'dc':
        train_dataloader = get_dc_dataloader(params)
    else:
        train_dataloader = get_dataloader(params)

    print('training dataloader created')

    if params.model == 'resnet34':
        optimizer = optim.Adam(params=model.parameters(), lr=params.lr)
    else:
        optimizer = optim.SGD(params=model.parameters(), lr=params.lr)
    print('optimizer created')

    print('starting training...')
    total_steps = 1
    start_time = time.time()
    for epoch in range(params.num_epochs):
        for imgs, classes in train_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate loss
            if params.model == 'resnet34':
                output, _ = model(imgs)
            else:
                output = model(imgs)
            loss = F.cross_entropy(output, classes)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAccuracy: {}'.format(
                        epoch + 1, total_steps, loss.item(), accuracy.item()
                    ))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # print out gradient values and parameter average values
            if total_steps % 1000 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # print and save the values
                    print('*' * 10)
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            avg_grad = torch.mean(param.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name), param.grad.cpu().numpy(), total_steps)
                        
                        if param.data is not None:
                            avg_weight = torch.mean(param.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
                            tbwriter.add_histogram('weight/{}'.format(name), param.data.cpu().numpy(), total_steps)

            total_steps += 1
        # save checkpoints
        checkpoint_path = os.path.join(checkpoint_dir, '{0}_states_e{1}.pkl'.format(model_id, epoch + 1))
        state = {
            'epoch' : epoch, 
            'total_steps' : total_steps,
            'optimizer' : optimizer.state_dict(),
            'model' : model.state_dict(),
            'seed' : params.seed
        }
        torch.save(state, checkpoint_path)

    end_time = time.time()
    time_taken = end_time - start_time

    # start model testing
    print("starting testing...")
    test_dataloader = get_dataloader(params, test=True)
    print('test dataloader created')
    accs = []
    for _ in range(params.num_runs):
        num_correct = 0
        num_samples = 0
        for imgs, classes in test_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            if params.model == 'resnet34':
                output, _ = model(imgs)
            else:
                output = model(imgs)
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == classes)
            num_correct += accuracy
            num_samples += preds.size(0)
        
        test_acc = float(num_correct) / float(num_samples) * 100
        accs.append(test_acc)
    
    # save model
    torch.save(model.state_dict(), model_path)
    # report metrics
    metric_file = get_model_metric_file(params)
    mean_test_acc = statistics.mean(accs)
    std_test_acc = statistics.stdev(accs)
    with open(metric_file, 'a') as f:
        f.write(
            'Model ID: {0}\nTraining Time: {1}\nNumber of Test Runs: {2}\nMean Test Acc: {3}\nStdev Test Acc: {4}\n\n'.format(
                model_id, time_taken, params.num_runs, mean_test_acc, std_test_acc)
        )
    f.close()



def class_wise_test_acc(model, params):
    _, checkpoint_dir = get_model_dump_paths(params)
    model_id = get_model_id(params)
    model_path = os.path.join(checkpoint_dir, '{0}.pt'.format(model_id))
    model.load_state_dict(torch.load(model_path))
    print('Model loaded from:{0}'.format(model_path))
    print('starting testing....')
    test_dataloader = get_dataloader(params, test=True)
    print('test dataloader created')
    class_wise_accs = np.zeros((params.num_classes, params.num_runs)) 
    for i in range(params.num_runs):
        class_wise_correct = [0] * params.num_classes
        class_wise_samples = [0] * params.num_classes
        for imgs, classes in test_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            if params.model == 'resnet34':
                output, _ = model(imgs)
            else:
                output = model(imgs)

            _, preds = torch.max(output, 1)
            for c in range(params.num_classes):
                class_wise_correct[c] += torch.sum((preds == classes) * (classes == c))
                class_wise_samples[c] += torch.sum(classes == c)
        
        for c in range(params.num_classes):
            class_wise_accs[c][i] = float(class_wise_correct[c]) / float(class_wise_samples[c]) * 100
            # print('Number of Points in class {0}: {1}'.format(c, class_wise_samples[c]))

    class_wise_accs = np.mean(class_wise_accs, axis=1)
    print(class_wise_accs)
    metric_file = get_model_metric_file(params)
    with open(metric_file, 'a') as f:
        f.write('Class Wise Test Accuracy\n')
        for idx, value in enumerate(class_wise_accs):
            f.write('Class:{0}\tAcc:{1}\n'.format(idx, value))
    f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="convnet", help="model type")
    
    # params for ConvNet
    parser.add_argument('--net_width', type=int, default=256)
    parser.add_argument('--net_depth', type=int, default=4)
    parser.add_argument('--net_act', type=str, default="leakyrelu")
    parser.add_argument('--net_norm', type=str, default="batchnorm")
    parser.add_argument('--net_pooling', type=str, default="avgpooling")

    # params for model training
    parser.add_argument('--num_runs', type=int, default=2, help="number of runs for model testing")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for model training")
    parser.add_argument('--lr', type=float, default=0.02, help="learning rate for model training")
    parser.add_argument('--seed', type=int, default=1234, help="seed to init torch")

    # params for data description
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--coreset', type=int, default=1)
    parser.add_argument('--algo_type', type=str, default="dc")
    parser.add_argument('--coverage_factor', type=int, default=30)
    parser.add_argument('--distribution_req', type=int, default=100)
    parser.add_argument('--partitions', type=int, default=10, help='number of partitions')
    parser.add_argument('--model_type', type=str, default='resnet-18')
    # parse all parameters
    params = parser.parse_args()

    if params.algo_type == 'greedyNC_fair':
        params.coverage_factor = 0
        params.algo_type = 'greedyNC'
        
    # toDo: add other datasets
    if params.dataset == 'cifar10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        params.num_classes = 10
    elif params.dataset == 'mnist':
        channel = 1
        im_size = (28, 28)
        params.num_classes = 10
        num_classes = 10
    elif params.dataset == 'lfw':
        params.num_classes = 2
        im_size = (128, 128)
        num_classes = 2
        channel = 3
    elif params.dataset == 'fashion-mnist':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        params.num_classes = 10


    tblog_path, checkpoint_dir = get_model_dump_paths(params)
    model_id = get_model_id(params)
    model_path = os.path.join(checkpoint_dir, '{0}.pt'.format(model_id))

    if not isfile(model_path):
    # TODO: add other networks
        model = None
        if params.model == 'alexnet':
            model = networks.AlexNet(num_classes)
        elif params.model == 'convnet':
            model = networks.ConvNet(channel, num_classes, params.net_width, params.net_depth, params.net_act, params.net_norm, params.net_pooling, im_size)
        elif params.model == 'resnet34':
            model = networks.resnet34()
        elif params.model == 'resnet50':
            model = networks.ResNet50(num_classes)

        model = model.to(device)
        model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)
        train_and_test(model, params)
        class_wise_test_acc(model, params)
