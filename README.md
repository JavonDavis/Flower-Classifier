# Flower Classifier using Pytorch

## TLDR

Example usage of predict script, given a checkpoint file named classifier_checkpoint

python predict.py -ip artichoke.jpg -cp classifier_checkpoint.pth -m cat_to_name.json --gpu=False -k 3

Example usage of train script with hidden units 25088, 4096 and 1000, assuming that there exists a flowers directory that has the training, validation and test images in the expected format.

python train.py -hu 25088 4096 1000

Use python predict.py -h or python train.py for details on more options

# Long story

In this project I built a classifier for flowers using models pretrained on imagenet.

## TODO:
Complete writeup of architecture and files
