# Advanced Convolutions applied to CIFAR10

In this repo, we test drive a variety of convolutions such as depth-wise separable convolutions, atrous/dilated convolutions to achieve a Fully Convolutional Neural network that achieves **85.75% ** test accuracy on CIFAR-10 dataset. 

Notebook used for training can be found [here]

## Model Summary

The final model has 4 Convolution blocks with 136,032 parameters.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 32, 32]             648
       BatchNorm2d-2           [-1, 24, 32, 32]              48
              ReLU-3           [-1, 24, 32, 32]               0
           Dropout-4           [-1, 24, 32, 32]               0
            Conv2d-5           [-1, 24, 32, 32]           5,184
       BatchNorm2d-6           [-1, 24, 32, 32]              48
              ReLU-7           [-1, 24, 32, 32]               0
           Dropout-8           [-1, 24, 32, 32]               0
            Conv2d-9           [-1, 24, 16, 16]           5,184
      BatchNorm2d-10           [-1, 24, 16, 16]              48
             ReLU-11           [-1, 24, 16, 16]               0
          Dropout-12           [-1, 24, 16, 16]               0
           Conv2d-13           [-1, 24, 16, 16]             576
           Conv2d-14           [-1, 24, 16, 16]           5,184
      BatchNorm2d-15           [-1, 24, 16, 16]              48
             ReLU-16           [-1, 24, 16, 16]               0
          Dropout-17           [-1, 24, 16, 16]               0
           Conv2d-18           [-1, 24, 16, 16]             216
      BatchNorm2d-19           [-1, 24, 16, 16]              48
             ReLU-20           [-1, 24, 16, 16]               0
          Dropout-21           [-1, 24, 16, 16]               0
           Conv2d-22             [-1, 24, 8, 8]             432
      BatchNorm2d-23             [-1, 24, 8, 8]              48
             ReLU-24             [-1, 24, 8, 8]               0
          Dropout-25             [-1, 24, 8, 8]               0
           Conv2d-26             [-1, 48, 8, 8]           1,152
           Conv2d-27             [-1, 48, 8, 8]          20,736
      BatchNorm2d-28             [-1, 48, 8, 8]              96
             ReLU-29             [-1, 48, 8, 8]               0
          Dropout-30             [-1, 48, 8, 8]               0
           Conv2d-31             [-1, 48, 8, 8]             432
      BatchNorm2d-32             [-1, 48, 8, 8]              96
             ReLU-33             [-1, 48, 8, 8]               0
          Dropout-34             [-1, 48, 8, 8]               0
           Conv2d-35             [-1, 96, 8, 8]           4,608
           Conv2d-36             [-1, 96, 8, 8]          82,944
      BatchNorm2d-37             [-1, 96, 8, 8]             192
             ReLU-38             [-1, 96, 8, 8]               0
          Dropout-39             [-1, 96, 8, 8]               0
           Conv2d-40             [-1, 96, 8, 8]           6,912
      BatchNorm2d-41             [-1, 96, 8, 8]             192
             ReLU-42             [-1, 96, 8, 8]               0
          Dropout-43             [-1, 96, 8, 8]               0
           Conv2d-44             [-1, 10, 8, 8]             960
        AvgPool2d-45             [-1, 10, 1, 1]               0
================================================================
Total params: 136,032
Trainable params: 136,032
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.79
Params size (MB): 0.52
Estimated Total Size (MB): 3.32
----------------------------------------------------------------
```
Equivalent of Max-Pooling is achieved by using following:
```
nn.Conv2d( in_channel, out_channel, kernel_size=3, padding=2, stride=2, dilation = 2)
```
Final Receptive field = 73

## Image Augmentation
[Albumentation Library](https://github.com/albumentations-team/albumentations) was primarily used for augmentation during training. Following schemes were chosen:
1. Horizontal Flip
2. ShiftScaleRotate
3. CoarseDropout with 1 hole of 16x16

Samples of Augmented images are below:

![Augmented Images](https://github.com/rajy4683/EVA6/blob/master/EVA6S7/imgs/S7EVA6_AugmentedImages.png)

## Training Logs
Training was performed with using both One Cycle LR policy and CyclicLR. Below are the accuracy and loss plots:
![Accuracy Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S7/imgs/S7EVA6_AccuracyPlots.png)
![Loss Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S7/imgs/S7EVA6_LossPlots.png)
![Classwise Accuracy](https://github.com/rajy4683/EVA6/blob/master/EVA6S7/imgs/S7EVA6_ClasswiseAccuracy.png)

```
Epoch: 84 Test set: Average loss: 0.4903, Accuracy: 83.310%
loss=0.678766667842865 batch_id=97: 100%|██████████| 98/98 [00:31<00:00,  3.07it/s]
Epoch: 85 Train set: Average loss: 0.7233, Accuracy: 74.366%, lr:0
Epoch: 85 Test set: Average loss: 0.4863, Accuracy: 83.540%
loss=0.7115206122398376 batch_id=97: 100%|██████████| 98/98 [00:31<00:00,  3.07it/s]
Epoch: 86 Train set: Average loss: 0.7133, Accuracy: 75.110%, lr:0
Epoch: 86 Test set: Average loss: 0.4877, Accuracy: 83.410%
loss=0.7396487593650818 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.06it/s]
Epoch: 87 Train set: Average loss: 0.7117, Accuracy: 75.006%, lr:0
Epoch: 87 Test set: Average loss: 0.4771, Accuracy: 83.730%
loss=0.7843133211135864 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.05it/s]
Epoch: 88 Train set: Average loss: 0.7034, Accuracy: 75.210%, lr:0
Epoch: 88 Test set: Average loss: 0.4730, Accuracy: 83.920%
loss=0.7435541152954102 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.06it/s]
Epoch: 89 Train set: Average loss: 0.6997, Accuracy: 75.308%, lr:0
Epoch: 89 Test set: Average loss: 0.4700, Accuracy: 83.950%
loss=0.693544864654541 batch_id=97: 100%|██████████| 98/98 [00:31<00:00,  3.07it/s]
Epoch: 90 Train set: Average loss: 0.6985, Accuracy: 75.440%, lr:0
Epoch: 90 Test set: Average loss: 0.4707, Accuracy: 83.980%
loss=0.7059686779975891 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.06it/s]
Epoch: 91 Train set: Average loss: 0.6910, Accuracy: 75.500%, lr:0
Epoch: 91 Test set: Average loss: 0.4744, Accuracy: 83.870%
loss=0.6287901997566223 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.06it/s]
Epoch: 92 Train set: Average loss: 0.6915, Accuracy: 75.698%, lr:0
Epoch: 92 Test set: Average loss: 0.4504, Accuracy: 84.780%
Model saved as Test Accuracy increased from  84.68  to  84.78
loss=0.7617901563644409 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.05it/s]
Epoch: 93 Train set: Average loss: 0.6792, Accuracy: 76.060%, lr:0
Epoch: 93 Test set: Average loss: 0.4327, Accuracy: 85.450%
Model saved as Test Accuracy increased from  84.78  to  85.45
Epoch: 94 Train set: Average loss: 0.6752, Accuracy: 76.422%, lr:0
Epoch: 94 Test set: Average loss: 0.4375, Accuracy: 85.060%
loss=0.6648911833763123 batch_id=97: 100%|██████████| 98/98 [00:31<00:00,  3.08it/s]
Epoch: 95 Train set: Average loss: 0.6714, Accuracy: 76.556%, lr:0
Epoch: 95 Test set: Average loss: 0.4295, Accuracy: 85.240%
loss=0.69306480884552 batch_id=97: 100%|██████████| 98/98 [00:31<00:00,  3.07it/s]
Epoch: 96 Train set: Average loss: 0.6558, Accuracy: 76.812%, lr:0
Epoch: 96 Test set: Average loss: 0.4299, Accuracy: 85.190%
loss=0.5959483981132507 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.05it/s]
Epoch: 97 Train set: Average loss: 0.6598, Accuracy: 76.764%, lr:0
Epoch: 97 Test set: Average loss: 0.4324, Accuracy: 85.120%
loss=0.6442763209342957 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.06it/s]
loss=0.7094371318817139 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.06it/s]
Epoch: 98 Train set: Average loss: 0.6504, Accuracy: 77.166%, lr:0
Epoch: 98 Test set: Average loss: 0.4216, Accuracy: 85.710%
Model saved as Test Accuracy increased from  85.45  to  85.71
loss=0.6451497077941895 batch_id=97: 100%|██████████| 98/98 [00:32<00:00,  3.05it/s]
Epoch: 99 Train set: Average loss: 0.6509, Accuracy: 77.190%, lr:0
Epoch: 99 Test set: Average loss: 0.4231, Accuracy: 85.580%
Epoch: 100 Train set: Average loss: 0.6412, Accuracy: 77.386%, lr:0
Epoch: 100 Test set: Average loss: 0.4182, Accuracy: 85.750%
Model saved as Test Accuracy increased from  85.71  to  85.75
Final model save path: /content/model_saves/model-26c43ba9ed.pt  best Accuracy: 85.75
```
## Links
1. Model and Training code is packaged in **[this repo](https://github.com/rajy4683/mini-Rekog)**
2. Notebook used for training can be found **[here](https://github.com/rajy4683/EVA6/blob/master/EVA6S7/S7EVA4_AdvancedConv_Final.ipynb)**

