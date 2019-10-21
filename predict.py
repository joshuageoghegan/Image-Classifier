#python ImageClassifier/predict.py --pathtoimage /home/workspace/ImageClassifier/flowers/train/102/image_08000.jpg --checkpoint /ImageClassifier/classifier_pth --topk 5 --catnames cat_to_name.json --gpu Y

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

import json

from train import get_input_args
#from train import main

in_arg = get_input_args()
with open('ImageClassifier/'+in_arg.catnames, 'r') as f:
    cat_to_name = json.load(f)

#import time

from PIL import Image

from collections import OrderedDict

from workspace_utils import active_session

from torchvision import datasets, transforms, models

import model_func
import argparse
from train import *

in_arg = get_input_args()
model_name = in_arg.arch
model = models[model_name]

def inference(in_arg, model, train_data, imagepath, cat_to_name):
    
    chkdir = in_arg.checkpoint_dir
        
    filepath = chkdir+'/classifier_pth'
    load_checkpoint(model, train_data, filepath)
    
    #process image for inference
    process_image(imagepath)
    
    #checking processing works
    #imshow(image)
    
    #predict top 5 probability
    predict(imagepath, model, train_data, cat_to_name)
    
    #sanity checking
    #SanityCheck(img, image_path, model)
    
train_data, validation_data, test_data = data_transform(in_arg)
imagepath = in_arg.pathtoimage
inference(in_arg, model, train_data, imagepath, cat_to_name)    