# Triton builder for ImageNet models

This repo allows to build Triton Inference Server docker image including model repository:

1) it downloads models you want to serve using triton

2) builds image that contains both models and triton (ready to run)


The model repository consists of torch models pretrained on the ImageNet dataset.

There is (should be) two types of models: classification model and feature extraction model (fe suffix of model name).

# Requirements

1. Ubuntu 20.04
2. Docker version 20.10.17, build 100c701
3. Python 3.9

For detailed python requirements see requirements.txt

# Available models

1. resnet18
2. resnet50_fe

# Usage

usage: cook_triton.py [-h] image_version models [models ...]

Example:
`bash
python cook_triton.py resnets resnet50_fe resnet18
`

# TODO

1. More models: add other models from torchvision (densenets, resnext, etc.)
2. Convert "Available models" list to table with descriptions
3. Do preprocessing on triton's side (use python backend)
