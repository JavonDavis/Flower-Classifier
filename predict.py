# Imports here
import os
import numpy as np
import torch
import torchvision
import sys
import argparse
import json

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

def load_checkpoint(path, gpu=True):
    if gpu:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')

    if checkpoint['model_name'] == 'vgg19':
        model = models.vgg19()
    elif checkpoint['model_name'] == 'vgg16':
        model = models.vgg16()
    else:
        raise Exception("Invalid model {}".format(checkpoint['model_name']))


    classifier = nn.Sequential(OrderedDict(
            [
                ('fc1', nn.Linear(25088, 4096)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(0.2)),
                ('fc2', nn.Linear(4096, 1000)),
                ('relu2', nn.ReLU()),
                ('dropout2', nn.Dropout(0.2)),
                ('fc3', nn.Linear(1000, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    width, height = 256, 256
    crop_width, crop_height = 224, 224
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # get the ratio that will get the shortest side to 256 then divide into both sides to maintain aspect ratio
    ratio = min(image.size[0], image.size[1])/256.

    image.thumbnail((image.size[0]/ratio, image.size[1]/ratio), Image.ANTIALIAS)

    left = (width - crop_width)//2
    top = (height - crop_height)//2
    right = (width + crop_width)//2
    bottom = (height + crop_height)//2

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)
    np_image = np_image/255.

    normalized_image = (np_image - mean) / std

    final_image = np.transpose(normalized_image, (2,0,1))
    return final_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = Image.open(image_path)
    image = process_image(image)

    image = torch.FloatTensor([image])
    model.eval()
    output = model.forward(Variable(image))
    ps = torch.exp(output).data.numpy()[0]

    positions = np.argsort(ps)[-topk:][::-1]
    topk_classes = [idx_to_class[position] for position in positions]
    topk_probs = ps[positions]

    return topk_probs, topk_classes

def is_gpu_available():
    return torch.cuda.is_available()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=lambda x: (str(x).lower() == 'true'), help='Whether to use a gpu if one is available or not', default=True)
    parser.add_argument('-ip', '--image_path', type=str, help='Path to image file', required=True)
    parser.add_argument('-cp', '--checkpoint_path', type=str, help='Path to checkpoint file', required=True)
    parser.add_argument('-k', '--topk', type=int, help='Top number of predictions to print', default=5)
    parser.add_argument('-m', '--mapping', type=str, help='Path to the file mapping class values to names', required=True)
    args = parser.parse_args()
    print(args)

    with open(args.mapping, 'r') as f:
        cat_to_name = json.load(f)

    args.gpu = args.gpu and is_gpu_available()
    loaded_model, checkpoint = load_checkpoint(args.checkpoint_path, args.gpu)

    # get index to class mapping
    idx_to_class = { v : k for k,v in checkpoint['class_to_idx'].items()}

    probs, classes = predict(args.image_path, loaded_model, args.gpu, args.topk)


    for i in range(args.topk):
        print('%s %.2f%%' % (cat_to_name[classes[i]], probs[i]*100))
