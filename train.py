# Imports here
import os
import numpy as np
import torch
import torchvision
import sys
import argparse

from PIL import Image

from collections import OrderedDict

from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import make_grid
from torch import optim
from torch import nn


# Training data paths
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
dirs = {'train': train_dir, 'val':valid_dir, 'test': test_dir}

def get_datasets():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(degrees=90),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    datasets = ['train', 'val', 'test']

    # Load the datasets with ImageFolder
    image_datasets = {x: ImageFolder(dirs[x], data_transforms[x]) for x in datasets}
    return image_datasets

def get_dataloaders(datasets, batch_size, should_shuffle=True, num_workers=2):
    return {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers) for x in datasets}

def get_pretrained_model(model_name='vgg19'):
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise Exception('Invalid model option {}'.format(model_name))
    return model

def get_classifier():

    classifier = []
    hidden_units = [int(x) for x in args.hidden_units]
    hidden_units_args = list(zip(hidden_units,hidden_units[1:]))
    hidden_units_args.append((hidden_units[-1], 102))
    for i, hidden_units_arg in enumerate(hidden_units_args):
        classifier.append(('fc'+str(i+1), nn.Linear(hidden_units_arg[0], hidden_units_arg[1])))
        if i == len(hidden_units_args) - 1:
            continue
        classifier.append(('relu'+str(i+1), nn.ReLU()))
        classifier.append(('dropout'+str(i+1), nn.Dropout(0.2)))

    classifier.append(('output', nn.LogSoftmax(dim=1)))
    classifier =  nn.Sequential(OrderedDict(classifier))
    return classifier

def is_gpu_available():
    return torch.cuda.is_available()

def get_optimizer(learning_rate, momentum):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=momentum)
    return criterion, optimizer

def feed_forward(model, dataloaders, datasets, is_training = True, gpu_available=False):
    dataset = 'val'
    if is_training:
        dataset = 'train'

    if is_training:
        model = model.train()
    else:
        model = model.eval()

    loss = 0
    num_correct = 0

    for inputs, labels in iter(dataloaders[dataset]):
        if gpu_available:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        if is_training:
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model.forward(inputs)
            ps = torch.exp(outputs).data
            c_loss = criterion(outputs, labels)
            c_loss.backward()
            optimizer.step()
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            outputs = model.forward(inputs)
            ps = torch.exp(outputs).data
            c_loss = criterion(outputs, labels)
        equals = (labels.data == ps.max(1)[1])
        loss += c_loss * inputs.size(0)
        num_correct += equals.type_as(torch.FloatTensor()).sum()

    dataset_size = len(datasets[dataset])
    epoch_loss = loss / dataset_size
    epoch_accuracy = num_correct / dataset_size

    print('%s Loss: %.4f Acc: %.4f' % (dataset, epoch_loss, epoch_accuracy))

def train(model, dataloaders, datasets, criterion, optimizer, num_epochs=5, gpu_available=False):

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for dataset in ['train', 'val']: feed_forward(model, dataloaders, datasets, dataset, gpu_available)
    return model

def save_checkpoint(model, model_name, datasets, optimizer, batch_size, num_epochs, num_classes=102):
    model.class_to_idx = datasets['train'].class_to_idx
    model.epochs = num_epochs
    checkpoint_path = 'classifier_checkpoint.pth'

    checkpoint = {'input_size': [3, 224, 224],
                  'batch_size': batch_size,
                  'output_size': num_classes,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'model_name': model_name,
                  'epoch': model.epochs}
    torch.save(checkpoint, checkpoint_path)

if __name__ == '__main__':

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, help='The batch size to use', default=32)
    parser.add_argument('-s', '--shuffle', type=bool, help='Whether to shuffle the data or not', default=True)
    parser.add_argument('-g', '--gpu', type=lambda x: (str(x).lower() == 'true'), help='Whether to use a gpu if one is available or not', default=True)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=5)
    parser.add_argument('-hu', '--hidden_units', nargs='+', help='Nuber of hidden units', required=True)
    parser.add_argument('-m', '--model', type=str, help='The pretrained model to use', default='vgg19', choices=['vgg19', 'vgg16'])
    parser.add_argument('-l', '--learning_rate', type=float, help='The Learning rate', default=0.01)
    parser.add_argument('-sm', '--momentum', type=float, help='The momentum for the SGD optimizer', default=0.9)
    args = parser.parse_args()
    print(args)



    image_datasets = get_datasets()
    print(len(image_datasets))
    dataloaders = get_dataloaders(image_datasets, args.batch_size, args.shuffle)
    print(len(dataloaders))
    model = get_pretrained_model(args.model)
    print(model)
    model.classifier = get_classifier()
    print(model)

    args.gpu = args.gpu and is_gpu_available()
    if args.gpu:
        model = model.cuda()

    criterion, optimizer = get_optimizer(args.learning_rate,args.momentum)
    train(model, dataloaders, image_datasets, criterion, optimizer, args.epochs, args.gpu)
    save_checkpoint(model, args.model, image_datasets, optimizer, args.batch_size, args.epochs)
