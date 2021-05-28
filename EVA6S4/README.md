# Small CNNs for Handwritten Digit recognition(MNIST)

## TOC

1. [Overview](#overview)
2. [Available Models](#available-models)
3. [Model Architecture](#model-architecture)
4. [Training Mechanism](#training-mechanism)
5. [Model Graphs](#model-graphs)
6. [References](#references)

## Overview:

 This repo contains Convolution models(in PyTorch) convolution-only model that attain accuracy of between 99-99.4% on Handwritten digits i.e [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

## Available Models:

Following models are available:

| Model Name       | Parameter Count | File Size | Max Validation Accuracy | Conv Blocks | Max Receptive Field | Model State_Dict | Training Notebook |
| :--------------- | --------------- | --------- | ----------------------- | ----------- | ------------------- | ---------------- | ----------------- |
| MNIST-Medium     | 7632            | 45KB      | 99.44                   | 3           |                     |                  |                   |
| MNIST-Small      | 5616            | 36KB      | 99.25                   | 2           |                     |                  |                   |
| MNIST-UltraSmall | 4464            | 33.2KB    | 99.00                   | 2           |                     |                  |                   |

## Model Architecture:

In general, the above models are structured as:

1. Two or three **convolution blocks **
2. Then AdaptiveAverage pooling 
3. Then final classification layer with Log Softmax.

Each **convolution block** has following layers:

1. At least 2 layers of 
   1. 3x3 convolutions
   2. RELU Activation
   3. Dropout
2. Max Pooling Layers (excluding last conv block)
3. Conv 1x1 layer (excluding last conv block)

Negative Log-likelihood is the loss function used.

All model definitions can be found **[here](https://github.com/rajy4683/EVA6/tree/master/EVA6S4/models/MNISTModels.py)**

## Training Mechanism

Hyperparameters used for every model can be found in this [link](). Standard training and test datasets were used.

For MNIST-Small and MNIST-Ultrasmall, additional augmentation such as Random rotation between (-5.0, 5.0) was applied during training.

## Model Graphs

[Netron](https://github.com/lutzroeder/netron) is an extremely handy tool to quickly visualize traced PyTorch models. Below are the visualizations for each of the models:

## References

Awesome course content from TSAI.
