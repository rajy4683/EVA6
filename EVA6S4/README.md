# Small CNNs for Handwritten Digit recognition(MNIST)

## TOC

1. [Overview](#overview)
2. [Available Models](#available-models)
3. [Model Architecture](#model-architecture)
4. [Training Mechanism](#training-mechanism)
5. [Model Graphs](#model-graphs)
6. [Training Logs and Visualizations](#training-logs-and-visualizations)
7. [References](#references)

## Overview:

 This repo contains Convolution models(in PyTorch) convolution-only model that attain accuracy of between 99-99.4% on Handwritten digits i.e [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

## Available Models:

Following models are available:

| Model Name       | Parameter Count | File Size | Max Validation Accuracy | Conv Blocks | Max Receptive Field | Model State_Dict                                             | Training Notebook                                            |
| :--------------- | --------------- | --------- | ----------------------- | ----------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MNIST-Medium     | 7288            | 41KB      | 99.44                   | 3           | 32                  | [Model Medium](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/saved_dict/mnist_medium.pth) | [Train Medium](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/MNIST_Medium_EVA6S4.ipynb) |
| MNIST-Small      | 5616            | 36KB      | 99.25                   | 2           | 20                  | [Model Small](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/saved_dict/mnist_small.pth) | [Train Small](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/MNIST_Small_EVA6S4.ipynb) |
| MNIST-UltraSmall | 4392            | 31KB      | 99.26                   | 2           | 35                  | [Model UltraSmall](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/saved_dict/mnist_ultrasmall.pth) | [Train UltraSmall](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/MNIST_UltraSmall_EVA6S4.ipynb) |

## Model Architecture:

In general, the above models are structured as:

1. Two or three **convolution blocks **
2. Then AdaptiveAverage pooling 
3. Then final classification layer(using 1x1 for converting to 10 classes) with Log Softmax.

Each **convolution block** has following layers:

1. At least 2 layers of 
   1. 3x3 convolutions
   2. RELU Activation
   3. Dropout
2. Max Pooling Layers (excluding last conv block)
3. Negative Log-likelihood is the loss function used.

All model definitions can be found **[here](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/models/MNISTModels.py)**

## Training Mechanism

Hyperparameters used for every model can be found in this [link](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/hyperparams.txt). Standard training and test datasets were used.

For MNIST-Small and MNIST-Ultrasmall, additional augmentation such as Random rotation between (-5.0, 5.0) was applied during training.

## Model Graphs

[Netron](https://github.com/lutzroeder/netron) is an extremely handy tool to quickly visualize traced PyTorch models. Below is the visualization for MNIST-Medium:

![MNIST Medium Model](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/pngs/MNIST_Medium_model.png)
Below are the links for MNIST-Small and MNIST-UltraSmall

[MNIST Small](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/pngs/MNIST_Small_model.png)

[MNIST UltraSmall](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/pngs/MNIST_UltraSmall_model.png)

## Training Logs and Visualizations

For MNIST-Medium below are the accuracy/loss curves:

![MNIST Medium](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/pngs/MNIST_Medium.PNG.jpg)

Below are the training logs:

```
loss=0.13210086524486542 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.34it/s]
Epoch: 1 Train set: Average loss: 0.0020, Accuracy: 55301/60000 (92.168%)
Epoch: 1 Test set: Average loss: 0.0657, Accuracy: 9774/10000 (97.740%)
loss=0.10309451818466187 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 42.73it/s]
Epoch: 2 Train set: Average loss: 0.0008, Accuracy: 58166/60000 (96.943%)
Epoch: 2 Test set: Average loss: 0.0473, Accuracy: 9854/10000 (98.540%)
loss=0.06141829118132591 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.43it/s]
Epoch: 3 Train set: Average loss: 0.0006, Accuracy: 58583/60000 (97.638%)
Epoch: 3 Test set: Average loss: 0.0311, Accuracy: 9895/10000 (98.950%)
loss=0.09592031687498093 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.54it/s]
Epoch: 4 Train set: Average loss: 0.0005, Accuracy: 58729/60000 (97.882%)
Epoch: 4 Test set: Average loss: 0.0332, Accuracy: 9892/10000 (98.920%)
loss=0.05666756629943848 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.28it/s]
Epoch: 5 Train set: Average loss: 0.0005, Accuracy: 58877/60000 (98.128%)
Epoch: 5 Test set: Average loss: 0.0294, Accuracy: 9913/10000 (99.130%)
loss=0.04356670379638672 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 42.93it/s]
Epoch: 6 Train set: Average loss: 0.0004, Accuracy: 58948/60000 (98.247%)
Epoch: 6 Test set: Average loss: 0.0228, Accuracy: 9926/10000 (99.260%)
loss=0.08044812828302383 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 41.96it/s]
Epoch: 7 Train set: Average loss: 0.0004, Accuracy: 58990/60000 (98.317%)
loss=0.06959786266088486 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 42.71it/s]
Epoch: 7 Test set: Average loss: 0.0233, Accuracy: 9925/10000 (99.250%)
Epoch: 8 Train set: Average loss: 0.0004, Accuracy: 59038/60000 (98.397%)
Epoch: 8 Test set: Average loss: 0.0250, Accuracy: 9926/10000 (99.260%)
loss=0.03342217579483986 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.52it/s]
Epoch: 9 Train set: Average loss: 0.0004, Accuracy: 59058/60000 (98.430%)
Epoch: 9 Test set: Average loss: 0.0262, Accuracy: 9921/10000 (99.210%)
loss=0.043970998376607895 batch_id=468: 100%|██████████| 469/469 [00:11<00:00, 42.10it/s]
Epoch: 10 Train set: Average loss: 0.0004, Accuracy: 59112/60000 (98.520%)
Epoch: 10 Test set: Average loss: 0.0225, Accuracy: 9927/10000 (99.270%)
loss=0.020948315039277077 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 44.23it/s]
Epoch: 11 Train set: Average loss: 0.0004, Accuracy: 59134/60000 (98.557%)
Epoch: 11 Test set: Average loss: 0.0286, Accuracy: 9909/10000 (99.090%)
loss=0.031056808307766914 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 43.60it/s]
Epoch: 12 Train set: Average loss: 0.0003, Accuracy: 59204/60000 (98.673%)
Epoch: 12 Test set: Average loss: 0.0228, Accuracy: 9929/10000 (99.290%)
loss=0.010536453686654568 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 43.85it/s]
Epoch: 13 Train set: Average loss: 0.0003, Accuracy: 59215/60000 (98.692%)
Epoch: 13 Test set: Average loss: 0.0192, Accuracy: 9944/10000 (99.440%)
loss=0.03376437723636627 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 42.74it/s]
Epoch: 14 Train set: Average loss: 0.0003, Accuracy: 59228/60000 (98.713%)
Epoch: 14 Test set: Average loss: 0.0258, Accuracy: 9923/10000 (99.230%)
loss=0.0177930761128664 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 42.86it/s]
Epoch: 15 Train set: Average loss: 0.0003, Accuracy: 59241/60000 (98.735%)
Epoch: 15 Test set: Average loss: 0.0195, Accuracy: 9941/10000 (99.410%)
loss=0.006927466485649347 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 43.49it/s]
Epoch: 16 Train set: Average loss: 0.0003, Accuracy: 59212/60000 (98.687%)
Epoch: 16 Test set: Average loss: 0.0221, Accuracy: 9926/10000 (99.260%)
loss=0.041628237813711166 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 43.51it/s]
Epoch: 17 Train set: Average loss: 0.0003, Accuracy: 59260/60000 (98.767%)
Epoch: 17 Test set: Average loss: 0.0199, Accuracy: 9940/10000 (99.400%)
loss=0.029962999746203423 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 43.18it/s]
Epoch: 18 Train set: Average loss: 0.0003, Accuracy: 59305/60000 (98.842%)
Epoch: 18 Test set: Average loss: 0.0188, Accuracy: 9940/10000 (99.400%)
loss=0.1513604074716568 batch_id=468: 100%|██████████| 469/469 [00:10<00:00, 43.46it/s]
Epoch: 19 Train set: Average loss: 0.0003, Accuracy: 59285/60000 (98.808%)
Epoch: 19 Test set: Average loss: 0.0204, Accuracy: 9932/10000 (99.320%)
```

Results are other models can be found in this [path](https://github.com/rajy4683/EVA6/tree/master/EVA6S4/pngs)

## References

Awesome course content from TSAI.
