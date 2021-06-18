# Advanced Convolutions applied to CIFAR10

In this repo, we test drive a variety of convolutions such as depth-wise separable convolutions, atrous/dilated convolutions to achieve a Fully Convolutional Neural network that can achieve ~85% test accuracy on CIFAR-10 dataset. 

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
![Augmented Images]()

## Training Logs
Training was performed with using both One Cycle LR policy and CyclicLR. Below are the accuracy and loss plots:
![]()
![]()

The model code and training functions are available in [this repo](https://github.com/rajy4683/mini-Rekog)

