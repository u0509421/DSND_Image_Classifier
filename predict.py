import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import seaborn as sns
import argparse

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p = drop_p)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


parser = argparse.ArgumentParser(description='Predict.py')

parser.add_argument('img_dir', type=str, default='flowers/test/100/image_07896.jpg')
parser.add_argument('checkpoint', type=str, default='checkpoint.pth')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names',  type=str, default='cat_to_name.json')
parser.add_argument('--gpu', type=str, default="gpu")

ap = parser.parse_args()

img_dir = ap.img_dir
path = ap.checkpoint
top_k = ap.top_k
mode = ap.gpu
cat_name = ap.category_names
    
    
import json

with open(cat_name, 'r') as f:
    cat_to_name = json.load(f)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16()
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13()
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

# Rebuild model
model_new, optimizer_new = load_checkpoint(path)
model_new

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(image)
    
    # Resize, crop, convert to array, normalized using transforms method
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    img_tensor = img_transforms(img)
    
    return img_tensor

# Process image
img = process_image(img_dir)
print(img.shape)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if mode == 'gpu':
        model.cuda()
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probability = F.softmax(output.data, dim=1)
    probs, classes = probability.topk(topk)
    probs = probs.detach().tolist()[0]
    classes = classes.detach().tolist()[0]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    labels = []
    for i in classes:
        labels.append(cat_to_name[idx_to_class[i]])
    
    return probs, labels

# Predict
probs, labels = predict(img_dir, model_new, top_k)


for i in range(top_k):
    print("{} with probability of {}".format(labels[i], probs[i]))

