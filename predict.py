#python ImageClassifier/predict.py --pathtoimage /home/workspace/ImageClassifier/flowers/train/102/image_08000.jpg --checkpoint /home/workspace/ImageClassifier/classifier.pth --topk 5 --category_names cat_to_name.json --gpu Y

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from utility_func import *

import json

from train import get_input_args

in_arg = get_input_args()
with open('ImageClassifier/'+in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

from PIL import Image

from collections import OrderedDict

from workspace_utils import active_session

from torchvision import datasets, transforms, models

import model_func
import argparse
from train import *

def main():
#def inference(in_arg, model, train_data, imagepath, cat_to_name):  

    in_arg = get_input_args()

    filepath = in_arg.checkpoint
          
    load_checkpoint(filepath, in_arg)
    
    #process image for inference
    process_image(imagepath)
    
    #checking processing works
    #imshow(image)
    
    #predict top 5 probability
    model = load_checkpoint(filepath, in_arg)
    
    #if in_arg.gpu == 'Y':
    #    model.to('cuda')
    #else:
    #    model.to('cpu')

    predict(imagepath, model, cat_to_name, in_arg)
    
    #sanity checking
    #SanityCheck(img, image_path, model)
    
#train_data, validation_data, test_data = data_transform(in_arg)
imagepath = in_arg.pathtoimage

if __name__ == '__main__': #need this otherwise if I create a function called main and import it, it will run instead.
    main()