import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

from collections import OrderedDict

from workspace_utils import active_session

import argparse

def get_input_args():
    
    # Creates parse 
    parser = argparse.ArgumentParser()
    
    #input args for train
    parser.add_argument('--data_dir', type=str, # default='/home/workspace/ImageClassifier/flowers',
                        help='path to folder of training files')
    parser.add_argument('--save_dir', type=str, default='home/workspace/ImageClassifier', 
                        help='path to folder to save checkpoint')
    parser.add_argument('--arch', type=str, default='densenet', 
                        help='chosen model - densenet or vgg')
    parser.add_argument('--learning_rate', type=float, default='0.0005', 
                        help='chose learning rate')
    parser.add_argument('--hidden_units', type=int, default='900', 
                        help='chose hidden unit count')
    parser.add_argument('--epochs', type=int, default='3', 
                        help='chose epoch count')
    parser.add_argument('--gpu', type=str, 
                        help='use GPU for training.')
    
    #input args for predict
    parser.add_argument('--pathtoimage', type=str, #default='/home/workspace/ImageClassifier/flowers/train/102/image_08000.jpg',
                        help='path to image for prediction')
    parser.add_argument('--checkpoint', type=str, default='classifier.pth',
                        help='path to checkpoint')
    parser.add_argument('--topk', type=int, default=5,
                        help='top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='mapping of categories to real names')
    
    # returns parsed argument collection - need to add try/catch script for incorrect values.
    return parser.parse_args()

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

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    return trainloader, validationloader, testloader

def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Process a PIL image for use in a PyTorch model    
    image = Image.open(imagepath)
    
    meanval = np.array([0.485, 0.456, 0.406])
    stdval = np.array([0.229, 0.224, 0.225])   

    #define transforms
    if image.size[1] > image.size[0]:
        image.thumbnail((256, 100000))
    else:
        image.thumbnail((100000, 256))

    width, height = image.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    image = image.crop((left, top, right, bottom))

    image = np.array(image)/255

    image = (image - meanval)/stdval

    image = image.transpose((2, 0, 1))

    return image

def predict(image_path, model, cat_to_name, in_arg, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Implement the code to predict the class from an image file
    #model.to('cpu')
    model.eval()

    image = process_image(image_path)

    #need to convert array to tensor
    im_tensor = torch.from_numpy(image).type(torch.FloatTensor)

    #first arg needs to be a batch size, so add 1 to tensor
    input = im_tensor.unsqueeze(0)

    output = model.forward(input)

    probability = torch.exp(output)

    #get top 5 probabilities
    top5_idx, top5_labels = probability.topk(5)

    #convert result to list before inverting    
    top5_labels = top5_labels.detach().numpy().tolist()[0]
    top5_probs = top5_idx.detach().numpy().tolist()[0]

    #invert dictionary
    inv_map = {v: k for k, v in model.class_to_idx.items()}

    top5_label = [inv_map[i] for i in top5_labels] 
    top5_flowers = [cat_to_name[inv_map[i]] for i in top5_labels]
    print("Top 5 labels: {}".format(top5_label)),
    print("Top 5 Probabilities: {}".format(top5_probs)),
    print("Top 5 Flowers: {}".format(top5_flowers)),
    print("Top 5 Index: {}".format(top5_idx))