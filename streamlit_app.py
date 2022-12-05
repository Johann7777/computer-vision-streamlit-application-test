import torch
import torch.nn as nn
#import torch.optim as optim
#import torch.multiprocessing as mulpr
#from torch.utils.data import DataLoader
#import numpy as np
#import torchvision
from torchvision import datasets, models, transforms
#import torch.multiprocessing as mp

#from datetime import datetime
#import matplotlib.pyplot as plt
#import time
import os
#import copy
import streamlit as st
#import pandas as pd
#from pathlib import Path
from PIL import Image
import io

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
modeldir = os.path.join(__location__, 'squeezenet.pt')
categorydir = os.path.join(__location__, 'ModelCategories')

uploaded_file = None

model_categories = os.listdir(categorydir)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class ModelCategory:
    def __init__(self, name: str, ):
        self.name = name

    def __str__(self):
        return f"{self.name}({self.age})"

def initialize_model(model_name, num_classes, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else models.ResNet18_Weights.DEFAULT)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(models.AlexNet_Weights.IMAGENET1K_V1 if use_pretrained else models.AlexNet_Weights.DEFAULT)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(models.VGG11_BN_Weights.IMAGENET1K_V1 if use_pretrained else models.VGG11_BN_Weights.DEFAULT)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(models.SqueezeNet1_0_Weights.IMAGENET1K_V1 if use_pretrained else models.SqueezeNet1_0_Weights.DEFAULT)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(models.Inception_V3_Weights.IMAGENET1K_V1 if use_pretrained else models.Inception_V3_Weights.DEFAULT)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)


def main():
    st.title('Image upload demo')
    load_image()

def load_model():
    model_ft, input_size = initialize_model("squeezenet", 10, True)
    checkpoint = torch.load(modeldir)
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    model_ft.eval()
    return model_ft


def load_labels():
    return classes

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], top5_prob[i].item())

def load_dataset_options(dir: str):
    option = st.selectbox('Please select the model type:', os.listdir(dir))
    filelocation = dir + r'\\' + option
    return option, filelocation
def load_model_options():
    architecture_dir = "None"
    #First, let the user select the type of model they would like to use
    model_type = st.selectbox('Please select what type of model you would like to use:', os.listdir(categorydir))
    if(model_type is not None):
        model_type_dir = categorydir + r'\\' + model_type

        #Next, let the user select which of the available models for the type they would like to use
        architecture = st.selectbox('Please select one of the available models:', os.listdir(model_type_dir))
        if(architecture is not None):
            architecture_dir = model_type_dir + r'/' + architecture

    return architecture, architecture_dir
def main():
    st.title('Pretrained model demo')
    selected_architecture, selected_architecture_dir = load_model_options()
    if(selected_architecture == "squeezenet.pt"):
        model = load_model()
        categories = load_labels()
        image = load_image()
        
        result = st.button('Run on image')
        if result:
            st.write('Calculating results...')
            predict(model, categories, image)
        
if __name__ == '__main__':
    main()