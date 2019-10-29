#python ImageClassifier/train.py --data_dir /home/workspace/ImageClassifier/flowers --save_dir /home/workspace/ImageClassifier --arch densenet --learning_rate 0.0005 --hidden_units 900 --epochs 3 --gpu Y
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session
from torchvision import datasets, transforms, models

import argparse
from model_func import *
from utility_func import *

vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

models = {'vgg': vgg16, 'densenet': densenet121}

def main():    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

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
    
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, train_data, in_arg, optimizer)
    
if __name__ == '__main__': #need this otherwise if I create a function called main and import it, it will run instead
    main()	