#python ImageClassifier/train.py --data_dir /home/workspace/ImageClassifier/flowers --checkpoint_dir /home/workspace/ImageClassifier --arch densenet --learning_rate 0.0005 --hidden_units 900 --epochs 3 --gpu Y

# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
####https://www.youtube.com/watch?v=XYUXFR5FSxI train.py -h to show help
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

#import time
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session
from torchvision import datasets, transforms, models

#import model_func
import argparse
from model_func import *
#from predict import *

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16, 'densenet': densenet121}

def main():

    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg 
    #check_command_line_arguments(in_arg)   
    
    model_name = in_arg.arch
    model = models[model_name]
    
    data_transform(in_arg)
    train_data, validation_data, test_data = data_transform(in_arg)
    
    data_load(train_data, validation_data, test_data)
    trainloader, validationloader, testloader = data_load(train_data, validation_data, test_data)
    
    #Freeze parameters so we don't backprop through them, as we are using a pretrained network
    freeze_param(model)
    
    #replace classifier in model
    classifier(model, in_arg)   

    #track loss and accuracy with validation data to determine best hyperparameters
    #validation(model, validationloader, criterion)    
    
    #train network
    deep_learn(model, trainloader, validationloader, in_arg)
    optimizer = deep_learn(model, trainloader, validationloader, in_arg)
    criterion = deep_learn(model, trainloader, validationloader, in_arg)
    
    #test using testing data
    testing(model, testloader, criterion)
    
    ####need to set directory for checkpoint with cmd line input, as well choosing architecture and hyperparameters????
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, train_data, in_arg, optimizer)   #doesn't like train data in class_to_idx
    
def data_transform(in_arg):
        #load datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),                                       
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    validation_transforms = transforms.Compose([transforms.CenterCrop(224),
                                                transforms.Resize(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    #Load the datasets with ImageFolder
    traindir = in_arg.data_dir + '/train'
    validdir = in_arg.data_dir + '/valid'
    testdir = in_arg.data_dir + '/test'
    
    train_data = datasets.ImageFolder(traindir, transform=train_transforms)
    validation_data = datasets.ImageFolder(validdir, transform=validation_transforms)
    test_data = datasets.ImageFolder(testdir, transform=test_transforms)
    
    return train_data, validation_data, test_data
    
def data_load(train_data, validation_data, test_data):    

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    return trainloader, validationloader, testloader

def get_input_args():
    
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates command line arguments args.dir for path to images files,
    # args.arch which Densenet model to use for classification.
    #data_dir = '/home/workspace/ImageClassifier/flowers'
    
    #input args for train
    parser.add_argument('--data_dir', type=str, default='/home/workspace/ImageClassifier/flowers',   #remove default so not optional???
                        help='path to folder of training files')
    #parser.add_argument('--train_dir', type=str, default=data_dir+'/train', 
    #                    help='path to folder of train images')
    #parser.add_argument('--valid_dir', type=str, default=data_dir+'/valid', 
    #                    help='path to folder of images')
    #parser.add_argument('--test_dir', type=str, default=data_dir+'/test', 
    #                    help='path to folder of images')
    parser.add_argument('--checkpoint_dir', type=str, default='home/workspace/ImageClassifier', 
                        help='path to folder to save checkpoint') 
    parser.add_argument('--arch', type=str, default='densenet', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default='0.0005', 
                        help='chose learning rate')
    parser.add_argument('--hidden_units', type=int, default='900', 
                        help='chose hidden unit count')
    parser.add_argument('--epochs', type=int, default='3', 
                        help='chose epoch count')
    parser.add_argument('--gpu', type=str, 
                        help='use GPU for training.')
    
    #input args for predict
    parser.add_argument('--pathtoimage', type=str, default='/home/workspace/ImageClassifier/flowers/train/102/image_08000.jpg',
                        help='path to image for prediction')
    parser.add_argument('--checkpoint', type=str, default='/ImageClassifier/classifier_pth',
                        help='path to checkpoint')
    parser.add_argument('--topk', type=int, default=5,
                        help='top K most likely classes')
    parser.add_argument('--catnames', type=str, default='cat_to_name.json',
                        help='mapping of categories to real names')
    # returns parsed argument collection - need to add try/catch script for incorrect values.
    return parser.parse_args()

    #for k, v in models.items():
        #if in_arg.arch != k:
            #print('Incorrect model type. Please choose from ''resnet'', ''alexnet'', ''vgg'', ''densenet''')
        #else:
            #parser.parse_args()

get_input_args()

if __name__ == '__main__': #need this otherwise if I create a function called main and import it, it will run instead. See https://www.datacamp.com/community/tutorials/functions-python-tutorial
    main()	