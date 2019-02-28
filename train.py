import argparse

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import seaborn as sns


parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('data_dir', type=str, default="./flowers/")
parser.add_argument('--save_dir', type=str,default="./checkpoint.pth")
parser.add_argument('--arch', type=str, default="vgg16")
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=120)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--gpu', type=str, default="gpu")

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
architecture = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
mode = args.gpu

data_dir = where
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=train_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloaders = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

if architecture == 'vgg16':
    model = models.vgg16(pretrained=True)
elif architecture == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    print("Only vgg16 and vgg13 are supported.")

print(model)

# Freeze parameters    
for param in model.parameters():
    param.requires_grad = False
    
# Define new classifier
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

arch = {"vgg16":25088,
        "vgg13":25088}

hidden_layers = [4096, 1024]
hidden_layers.append(hidden_units)
classifier = Network(arch[architecture], 102, hidden_layers, drop_p=0.5)
model.classifier = classifier

          
# Train network
def validation(model, dataloader, criterion):
    test_loss = 0
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = round(correct/total * 100, 2)
            
        return test_loss, accuracy
    
def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    
    # change to cuda
    if device == 'gpu':
          model.to('cuda')
    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            _,predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/print_every),
                      "Test Accuracy: {:.3f}".format(accuracy))
                
                running_loss = 0
                # Make sure training is back on
                model.train()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
          
do_deep_learning(model, trainloaders, validloaders, epochs, 40, criterion, optimizer, mode)
          
# TODO: Do validation on the test set
_, accuracy = validation(model, testloaders, criterion)
print("Test Accuracy: {:.2f}".format(accuracy))

model.cpu
checkpoint = {'arch': architecture,
              'model_state_dict': model.state_dict(),
              'epochs': epochs,
              'class_to_idx': train_data.class_to_idx,
              'learning_rate': lr,
              'optimizer_state_dict': optimizer.state_dict(),
              'classifier': model.classifier}
print(checkpoint)
torch.save(checkpoint, path)