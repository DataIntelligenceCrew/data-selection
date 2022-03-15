"""
Implementation of AlexNET, from paper:
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""


import time 
from inspect import Parameter
import os
from matplotlib.pyplot import hist
from numpy import pad
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from tensorboardX import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# directories for models
INPUT_ROOT_DIR = '/localdisk3/data-selection/model-data/'
TEST_IMG_DIR = INPUT_ROOT_DIR + 'test/'
OUTPUT_DIR = '/localdisk3/data-selection/model-checkpoints/'
# directory where metrics for experiments are stored
RUNS_DIR = '../runs/'

# model parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR = 0.0001
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227 
NUM_CLASSES = 10 # CIFAR-10
DEVICE_IDS = [0, 1, 2, 3] # GPUs to use




def get_metric_filename(args):
    if args.coreset == 1:
        return "metrics_full_data_ml_" + str(args.sample_weight) + ".txt"
    return "metrics_" + str(args.coverage_factor) + "_" + str(args.sample_weight) + "_" + str(args.distribution_req) + "_" + str(args.composable) + '.txt'

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 9216, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=num_classes)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    


def train(args, CHECKPOINT_DIR, LOG_DIR):
    seed = torch.initial_seed()
    print('used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('tensorboardX summary writer created')

    # init model
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)
    print(model)
    print('model initialized')

    training_data_loc = INPUT_ROOT_DIR + "sampled_" + str(args.sample_weight) +"/"

    if args.coreset == 0:
        tp = "coreset"
    else:
        tp = "full_data"

    if tp != "full_data":
        training_data_loc += str(args.distribution_req) + "/" + tp
    else:
        training_data_loc += tp
    training_data_loc += str(args.composable) + "/train"
    print(training_data_loc)
    # create data loader
    train_dataset = datasets.ImageFolder(training_data_loc, transforms.Compose([
        transforms.Resize((227,227)), 
        transforms.RandomHorizontalFlip(p=0.7), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    print('training dataset created')

    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE
    )
    print('training dataloader created')

    # create optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LR)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    print('optimizer created')

    # multiply LR by 0.1 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR scheduler created')

    # start training
    print('starting training...')
    total_steps = 1
    # with torch.autograd.set_detect_anomaly(True):
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in train_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
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
            if total_steps % 100 == 0:
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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch' : epoch,
            'total_steps' : total_steps,
            'optimizer' : optimizer.state_dict(),
            'model' : model.state_dict(),
            'seed' : seed,
        }
        torch.save(state, checkpoint_path)
    
    end_time = time.time()
    # save model
    model_path = os.path.join(CHECKPOINT_DIR, 'alexnet_dist{0}_cf{1}_core{2}.pt'.format(args.distribution_req, args.coverage_factor, args.coreset))
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(RUNS_DIR, get_metric_filename(args)), "a") as f:
        f.write(
            "Time Taken to Train AlexNet Model: {0}\n".format((end_time - start_time))
        )
    f.close()

def test_model(args, CHECKPOINT_DIR):
    print('starting testing')
    # load model
    model_save_loc = [os.path.join(CHECKPOINT_DIR, _) for _ in os.listdir(CHECKPOINT_DIR) if _.endswith('.pt')]
    model_path = model_save_loc[0]
    # model_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e90.pkl')
    # state = torch.load(model_path)
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)
    # model.load_state_dict(state['model'])
    model.load_state_dict(torch.load(model_path))
    print('model loaded from:{0}'.format(model_path))
    # create data loader
    test_dataset = datasets.ImageFolder(TEST_IMG_DIR, transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    print('testing dataset created')

    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE
    )
    print('test dataloader created')
    num_correct = 0
    num_samples = 0
    for imgs, classes in test_dataloader:
        imgs, classes = imgs.to(device), classes.to(device)
        output = model(imgs)
        _, preds = torch.max(output, 1)
        accuracy = torch.sum(preds == classes)
        num_correct += accuracy
        num_samples += preds.size(0)
    
    print(f'test accuracy: {float(num_correct) / float(num_samples) * 100:.2f}')
    print()
    with open(os.path.join(RUNS_DIR, get_metric_filename(args)), "a") as f:
        f.write(
            "Test Accuracy AlexNet Model: {0}\n".format((float(num_correct) / float(num_samples)))
        )
    f.close()


def main(args):
    output_dir = OUTPUT_DIR + "sampled_" + str(args.sample_weight) +"/"

    if args.coreset == 0:
        tp = "coreset"
    else:
        tp = "full_data"

    if tp != "full_data":
        output_dir += str(args.distribution_req) + "/" + tp + str(args.composable)
    else:
        output_dir += tp 


    LOG_DIR = output_dir + 'tblogs/'
    CHECKPOINT_DIR = output_dir + 'models/'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if args.train == 0:
        train(args, CHECKPOINT_DIR, LOG_DIR)
    else:
        test_model(args, CHECKPOINT_DIR)





    