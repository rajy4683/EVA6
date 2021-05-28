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
| MNIST-Medium     | 7632            | 45KB      | 99.44                   | 3           | 32                  | [Model Medium](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/saved_dict/mnist_medium.pth) | [Train Medium](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/MNIST_Medium_EVA6S4.ipynb) |
| MNIST-Small      | 5616            | 36KB      | 99.25                   | 2           | 20                  | [Model Small](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/saved_dict/mnist_small.pth) | [Train Small](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/MNIST_Small_EVA6S4.ipynb) |
| MNIST-UltraSmall | 4464            | 33.2KB    | 99.00                   | 2           | 35                  | [Model UltraSmall](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/saved_dict/mnist_ultrasmall.pth) | [Train UltraSmall](https://github.com/rajy4683/EVA6/blob/master/EVA6S4/notebooks/MNIST_UltraSmall_EVA6S4.ipynb) |

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
loss=0.03157374635338783 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 62.50it/s]
loss=0.06512744724750519 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.92it/s]
Epoch: 3 Test set: Average loss: 0.0302, Accuracy: 9903/10000 (99.030%)
Epoch: 4 Train set: Average loss: 0.0012, Accuracy: 58646/60000 (97.743%)
loss=0.07279963046312332 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 62.40it/s]
Epoch: 4 Test set: Average loss: 0.0305, Accuracy: 9905/10000 (99.050%)
Epoch: 5 Train set: Average loss: 0.0011, Accuracy: 58763/60000 (97.938%)
loss=0.008938002400100231 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.59it/s]
Epoch: 5 Test set: Average loss: 0.0306, Accuracy: 9902/10000 (99.020%)
Epoch: 6 Train set: Average loss: 0.0009, Accuracy: 58903/60000 (98.172%)
loss=0.09788103401660919 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 59.07it/s]
Epoch: 6 Test set: Average loss: 0.0275, Accuracy: 9914/10000 (99.140%)
Epoch: 7 Train set: Average loss: 0.0009, Accuracy: 58918/60000 (98.197%)
loss=0.08261961489915848 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.32it/s]
Epoch: 7 Test set: Average loss: 0.0283, Accuracy: 9910/10000 (99.100%)
Epoch: 8 Train set: Average loss: 0.0008, Accuracy: 59030/60000 (98.383%)
loss=0.2145431488752365 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.15it/s]
Epoch: 8 Test set: Average loss: 0.0257, Accuracy: 9914/10000 (99.140%)
Epoch: 9 Train set: Average loss: 0.0008, Accuracy: 59019/60000 (98.365%)
loss=0.006780568510293961 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.00it/s]
Epoch: 9 Test set: Average loss: 0.0259, Accuracy: 9923/10000 (99.230%)
Epoch: 10 Train set: Average loss: 0.0008, Accuracy: 59040/60000 (98.400%)
loss=0.025659779086709023 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 59.74it/s]
Epoch: 10 Test set: Average loss: 0.0246, Accuracy: 9930/10000 (99.300%)
Epoch: 11 Train set: Average loss: 0.0008, Accuracy: 59125/60000 (98.542%)
loss=0.11572294682264328 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.90it/s]
Epoch: 11 Test set: Average loss: 0.0234, Accuracy: 9934/10000 (99.340%)
Epoch: 12 Train set: Average loss: 0.0008, Accuracy: 59090/60000 (98.483%)
loss=0.13506565988063812 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.87it/s]
Epoch: 12 Test set: Average loss: 0.0247, Accuracy: 9931/10000 (99.310%)
Epoch: 13 Train set: Average loss: 0.0007, Accuracy: 59144/60000 (98.573%)
loss=0.019705627113580704 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.52it/s]
Epoch: 13 Test set: Average loss: 0.0232, Accuracy: 9940/10000 (99.400%)
Epoch: 14 Train set: Average loss: 0.0007, Accuracy: 59181/60000 (98.635%)
loss=0.00933801755309105 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.99it/s]
Epoch: 14 Test set: Average loss: 0.0203, Accuracy: 9944/10000 (99.440%)
Epoch: 15 Train set: Average loss: 0.0007, Accuracy: 59160/60000 (98.600%)
loss=0.06273490190505981 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.74it/s]
Epoch: 15 Test set: Average loss: 0.0235, Accuracy: 9926/10000 (99.260%)
Epoch: 16 Train set: Average loss: 0.0007, Accuracy: 59191/60000 (98.652%)
loss=0.07527562230825424 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 61.01it/s]
Epoch: 16 Test set: Average loss: 0.0252, Accuracy: 9928/10000 (99.280%)
Epoch: 17 Train set: Average loss: 0.0007, Accuracy: 59217/60000 (98.695%)
loss=0.016707701608538628 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.62it/s]
Epoch: 17 Test set: Average loss: 0.0201, Accuracy: 9941/10000 (99.410%)
Epoch: 18 Train set: Average loss: 0.0006, Accuracy: 59252/60000 (98.753%)
loss=0.020643873140215874 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.46it/s]
Epoch: 18 Test set: Average loss: 0.0212, Accuracy: 9934/10000 (99.340%)
Epoch: 19 Train set: Average loss: 0.0006, Accuracy: 59230/60000 (98.717%)
```

Results are other models can be found in this [path](https://github.com/rajy4683/EVA6/tree/master/EVA6S4/pngs)

## References

Awesome course content from TSAI.
