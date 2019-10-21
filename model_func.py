#model functions and JOSH

# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

#import time

from PIL import Image

from collections import OrderedDict

from workspace_utils import active_session

def freeze_param(model):

    #model = models.densenet121(pretrained=True)
    #model

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
    #train classifier
    #if in_arg.gpu == 'Y': #how to test if using GPU on Udacity????
        #model.to('cuda')
    #else:
        #model.to('cpu')
        
    model.to('cuda') 
    
    epoch = in_arg.epochs
    lrate = in_arg.learning_rate

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lrate)
    
    #deep learn
    with active_session():
        epochs = epoch
        print_every = 10
        steps = 0
        
        for e in range(epochs):
            model.train()
            #start = time.time()
            running_loss = 0
            for images, labels in iter(trainloader):
                steps += 1
                images, labels = images.to('cuda'), labels.to('cuda')
                #images.resize_(images.size()[0], 1024)

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
                    model.to('cpu')
    return criterion

def testing(model, testloader, criterion):
    model.eval()

    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion)

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))   

def save_checkpoint(model, train_data, in_arg, optimizer):

    model.class_to_idx = train_data.class_to_idx
    
    chkdir = in_arg.checkpoint_dir

    checkpt = {'arch': model,
               'classifier': classifier,
               'input_size': 1024,
               'output_size': 102,
               'batch_size': 64,
               'epoch': 3,
               'class_to_idx': model.class_to_idx,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpt, chkdir+'/classifier_pth')
    #message to say successful save, or incorrect path?

def load_checkpoint(model, train_data, filepath):
    checkpt = torch.load(filepath, map_location = 'cpu')
    #model = models.densenet121(pretrained = True)

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
    model.class_to_idx = train_data.class_to_idx

    model.load_state_dict(checkpt['model_state_dict'])

def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model    
    image = Image.open(imagepath)
    
    meanval = np.array([0.485, 0.456, 0.406])
    stdval = np.array([0.229, 0.224, 0.225])   

    # define transforms
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
'''
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
'''
def predict(image_path, model, train_data, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    model.to('cpu')
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
    return top5_label, top5_probs, top5_flowers, top5_idx
'''
def SanityCheck(img, image_path, model):

    fig = plt.figure(figsize = [5, 10])
    ax = plt.subplot(2,1,1)

    #remove ticks and labels from image axis
    plt.xticks([])
    plt.yticks([])

    #Extract flower number to then map to flower name for title.
    Flower_No = image_path.split("/")[2]
    Flower_Name = cat_to_name[Flower_No]
    fig.suptitle(Flower_Name, fontsize=20)
    img_processed = process_image(img)

    imshow(img_processed, ax, title = Flower_Name)
    #get result from previous step to use as data    
    lbl, prob, flower, idx = predict(img, model)

    plt.subplot(2,1,2)
    base_color = sb.color_palette()[0]
    sb.barplot(x = prob, y = flower, color = base_color)
'''