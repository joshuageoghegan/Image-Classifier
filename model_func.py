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

from utility_func import *

def freeze_param(model):

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False #already using trained model, so don't calculate gradients
        
    return(model)

#replace classifier in model
def classifier(model, in_arg):
    
    hidden = in_arg.hidden_units
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, hidden)), #input to hidden1
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(0.3)),
                              ('fc2', nn.Linear(hidden, 102)), #hidden1 to output
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

def validation(model, loader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in loader:

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def deep_learn(model, trainloader, validationloader, in_arg):
    
    if in_arg.gpu == 'Y':
        model.to('cuda')
    else:
        model.to('cpu')
        
    epoch = in_arg.epochs
    lrate = in_arg.learning_rate

    criterion = nn.NLLLoss()

    #Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lrate)
    
    #deep learn
    with active_session():
        epochs = epoch
        print_every = 10
        steps = 0
        
        for e in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in iter(trainloader):
                steps += 1
                images, labels = images.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward pass
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step() #updates the weights

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval() #turn dropout off

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, validationloader, criterion)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(test_loss/len(validationloader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                    running_loss = 0                    

                    model.train() #turn dropout back on                    
    return criterion
    return model

def testing(model, testloader, criterion):
    model.eval()

    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion)

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))   

def save_checkpoint(model, train_data, in_arg, optimizer):

    model.class_to_idx = train_data.class_to_idx
    
    chkdir = in_arg.save_dir

    checkpt = {'arch': model,
               'classifier': classifier,
               'input_size': 1024,
               'output_size': 102,
               'batch_size': 64,
               'epoch': 3,
               'model': model,
               'class_to_idx': model.class_to_idx,
               'model_state_dict': model.state_dict(),
               'class_to_idx': model.class_to_idx,
               'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpt, chkdir+'/classifier.pth')

def load_checkpoint(filepath, in_arg):
    checkpt = torch.load(filepath, map_location = 'cpu')
    
    model = checkpt['model']
    
    for param in model.parameters():
        param.requires_grad = False #already using trained model, so don't calculate gradients

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 900)), #input to hidden1
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.3)),
                          ('fc2', nn.Linear(900, 102)), #hidden1 to output
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    model.load_state_dict(checkpt['model_state_dict'])
    return model