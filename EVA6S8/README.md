# ResNet Models on CIFAR10

In this repo, we evaluate modified ResNet18 model (using LayerNorm instead of BatchNormalization) to achieve **82.99% ** test accuracy on CIFAR-10 dataset. 

Notebook used for training can be found [here](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/EVA6S8_SingleScript.ipynb)

The main repo used for training the models can be found [here](https://github.com/rajy4683/mini-Rekog)

## Model Summary

The final model has 4 Convolution blocks with 11,173,962 parameters.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
      BasicLNBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
     BasicLNBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
        GroupNorm-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
        GroupNorm-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
        GroupNorm-18          [-1, 128, 16, 16]             256
     BasicLNBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
        GroupNorm-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
        GroupNorm-23          [-1, 128, 16, 16]             256
     BasicLNBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
        GroupNorm-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
        GroupNorm-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
        GroupNorm-30            [-1, 256, 8, 8]             512
     BasicLNBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
        GroupNorm-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
        GroupNorm-35            [-1, 256, 8, 8]             512
     BasicLNBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
        GroupNorm-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
        GroupNorm-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
        GroupNorm-42            [-1, 512, 4, 4]           1,024
     BasicLNBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
        GroupNorm-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
        GroupNorm-47            [-1, 512, 4, 4]           1,024
     BasicLNBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```
## Image Augmentation
[Albumentation Library](https://github.com/albumentations-team/albumentations) was primarily used for augmentation during training. Following schemes were chosen:
1. RandomCrop and Pad
2. CutOut(16x16)
3. Rotate(+/- 5 deg)

Samples of Augmented images are below:

![Augmented Images](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/imgs/EVA6S8_SampleAugmentation.png)

## GradCam and Misclassified Images
For debugging CNNs, GradCam is an important tool. Below are samples of misclassified images and GradCam applied to understand where the model tends to focus:
![Misclassified Images](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/imgs/EVA6S8_Misclassified.png)

| GradCam at Layer 2 (16x16)                                   | GradCam at Layer 3 (8x8)                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![GradCam Images at Layer 2](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/imgs/EVA6S8_GradCamL16.png) | ![GradCam Images at Layer 3](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/imgs/EVA6S8_GradCamL8.png) |

## Training Logs

Training was performed with using ReduceLROnPlateau for 40 epochs. Below are the accuracy and loss plots:
![Accuracy Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/imgs/EVA6S8_Accuracy.png)
![Loss Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/imgs/EVA6S8_LossPlot.png)

```
loss=2.3037612438201904 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 1 Train set: Average loss: 2.8804, Accuracy: 10.074%, lr:0
Epoch: 1 Test set: Average loss: 2.3072, Accuracy: 10.310%
Model saved as Test Accuracy increased from 0.0 to 10.31 at epoch 1
loss=2.3035237789154053 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 2 Train set: Average loss: 2.3062, Accuracy: 9.890%, lr:0
Epoch: 2 Test set: Average loss: 2.3041, Accuracy: 10.800%
Model saved as Test Accuracy increased from 10.31 to 10.8 at epoch 2
loss=2.313493490219116 batch_id=97: 100% 98/98 [00:34<00:00,  2.88it/s]

Epoch: 3 Train set: Average loss: 2.3058, Accuracy: 10.106%, lr:0
Epoch: 3 Test set: Average loss: 2.3042, Accuracy: 9.880%
loss=2.3058316707611084 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 4 Train set: Average loss: 2.3052, Accuracy: 10.240%, lr:0
Epoch: 4 Test set: Average loss: 2.3001, Accuracy: 10.230%
loss=2.2958950996398926 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 5 Train set: Average loss: 2.3027, Accuracy: 10.542%, lr:0
Epoch: 5 Test set: Average loss: 2.2948, Accuracy: 10.490%
loss=2.0815436840057373 batch_id=97: 100% 98/98 [00:34<00:00,  2.88it/s]

Epoch: 6 Train set: Average loss: 2.1797, Accuracy: 16.684%, lr:0
Epoch: 6 Test set: Average loss: 2.1483, Accuracy: 16.520%
Model saved as Test Accuracy increased from 10.8 to 16.52 at epoch 6
loss=1.9222352504730225 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 7 Train set: Average loss: 2.0134, Accuracy: 19.994%, lr:0
Epoch: 7 Test set: Average loss: 2.1415, Accuracy: 20.540%
Model saved as Test Accuracy increased from 16.52 to 20.54 at epoch 7
loss=1.7183611392974854 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 8 Train set: Average loss: 1.8725, Accuracy: 25.986%, lr:0
Epoch: 8 Test set: Average loss: 1.8615, Accuracy: 30.390%
Model saved as Test Accuracy increased from 20.54 to 30.39 at epoch 8
loss=1.6121582984924316 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 9 Train set: Average loss: 1.7249, Accuracy: 34.224%, lr:0
Epoch: 9 Test set: Average loss: 1.8114, Accuracy: 35.410%
Model saved as Test Accuracy increased from 30.39 to 35.41 at epoch 9
loss=1.4820207357406616 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 10 Train set: Average loss: 1.6040, Accuracy: 39.914%, lr:0
Epoch: 10 Test set: Average loss: 1.7332, Accuracy: 37.820%
Model saved as Test Accuracy increased from 35.41 to 37.82 at epoch 10
loss=1.3232345581054688 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 11 Train set: Average loss: 1.4963, Accuracy: 44.728%, lr:0
Epoch: 11 Test set: Average loss: 1.6788, Accuracy: 40.510%
Model saved as Test Accuracy increased from 37.82 to 40.51 at epoch 11
loss=1.4273918867111206 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 12 Train set: Average loss: 1.3912, Accuracy: 48.984%, lr:0
Epoch: 12 Test set: Average loss: 1.5911, Accuracy: 44.430%
Model saved as Test Accuracy increased from 40.51 to 44.43 at epoch 12
loss=1.2158440351486206 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 13 Train set: Average loss: 1.3015, Accuracy: 52.160%, lr:0
Epoch: 13 Test set: Average loss: 1.4621, Accuracy: 48.920%
Model saved as Test Accuracy increased from 44.43 to 48.92 at epoch 13
loss=1.1579006910324097 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 14 Train set: Average loss: 1.2113, Accuracy: 55.954%, lr:0
Epoch: 14 Test set: Average loss: 1.3356, Accuracy: 53.390%
Model saved as Test Accuracy increased from 48.92 to 53.39 at epoch 14
loss=1.1443021297454834 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 15 Train set: Average loss: 1.1358, Accuracy: 58.892%, lr:0
Epoch: 15 Test set: Average loss: 1.3433, Accuracy: 52.260%
loss=1.015799880027771 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 16 Train set: Average loss: 1.0504, Accuracy: 62.366%, lr:0
Epoch: 16 Test set: Average loss: 1.1699, Accuracy: 58.700%
Model saved as Test Accuracy increased from 53.39 to 58.7 at epoch 16
loss=1.0275461673736572 batch_id=97: 100% 98/98 [00:34<00:00,  2.88it/s]

Epoch: 17 Train set: Average loss: 0.9721, Accuracy: 65.284%, lr:0
Epoch: 17 Test set: Average loss: 1.1532, Accuracy: 60.150%
Model saved as Test Accuracy increased from 58.7 to 60.15 at epoch 17
loss=0.9345986247062683 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 18 Train set: Average loss: 0.9073, Accuracy: 67.534%, lr:0
Epoch: 18 Test set: Average loss: 1.0013, Accuracy: 65.140%
Model saved as Test Accuracy increased from 60.15 to 65.14 at epoch 18
loss=0.8993415832519531 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 19 Train set: Average loss: 0.8647, Accuracy: 69.326%, lr:0
Epoch: 19 Test set: Average loss: 0.9591, Accuracy: 66.860%
Model saved as Test Accuracy increased from 65.14 to 66.86 at epoch 19
loss=0.8127729296684265 batch_id=97: 100% 98/98 [00:34<00:00,  2.88it/s]

Epoch: 20 Train set: Average loss: 0.8037, Accuracy: 71.554%, lr:0
Epoch: 20 Test set: Average loss: 0.9943, Accuracy: 66.430%
loss=0.752876341342926 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 21 Train set: Average loss: 0.7634, Accuracy: 73.100%, lr:0
Epoch: 21 Test set: Average loss: 0.8480, Accuracy: 70.660%
Model saved as Test Accuracy increased from 66.86 to 70.66 at epoch 21
loss=0.7688126564025879 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 22 Train set: Average loss: 0.7214, Accuracy: 74.722%, lr:0
Epoch: 22 Test set: Average loss: 0.8157, Accuracy: 71.920%
Model saved as Test Accuracy increased from 70.66 to 71.92 at epoch 22
loss=0.6616466045379639 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]
  0% 0/98 [00:00<?, ?it/s]
Epoch: 23 Train set: Average loss: 0.6869, Accuracy: 75.942%, lr:0
Epoch: 23 Test set: Average loss: 0.8230, Accuracy: 71.640%
loss=0.7347068190574646 batch_id=97: 100% 98/98 [00:34<00:00,  2.88it/s]

Epoch: 24 Train set: Average loss: 0.6636, Accuracy: 76.678%, lr:0
Epoch: 24 Test set: Average loss: 0.7900, Accuracy: 72.710%
Model saved as Test Accuracy increased from 71.92 to 72.71 at epoch 24
loss=0.6522471308708191 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 25 Train set: Average loss: 0.6298, Accuracy: 78.000%, lr:0
Epoch: 25 Test set: Average loss: 0.7421, Accuracy: 74.770%
Model saved as Test Accuracy increased from 72.71 to 74.77 at epoch 25
loss=0.6558800339698792 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 26 Train set: Average loss: 0.6011, Accuracy: 78.756%, lr:0
Epoch: 26 Test set: Average loss: 0.6916, Accuracy: 76.100%
Model saved as Test Accuracy increased from 74.77 to 76.1 at epoch 26
loss=0.6644287705421448 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 27 Train set: Average loss: 0.5791, Accuracy: 79.730%, lr:0
Epoch: 27 Test set: Average loss: 0.7256, Accuracy: 75.220%
loss=0.64204341173172 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 28 Train set: Average loss: 0.5530, Accuracy: 80.760%, lr:0
Epoch: 28 Test set: Average loss: 0.7354, Accuracy: 74.290%
loss=0.4992201328277588 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 29 Train set: Average loss: 0.5288, Accuracy: 81.358%, lr:0
Epoch: 29 Test set: Average loss: 0.6452, Accuracy: 78.370%
Model saved as Test Accuracy increased from 76.1 to 78.37 at epoch 29
loss=0.6473566889762878 batch_id=97: 100% 98/98 [00:34<00:00,  2.88it/s]

Epoch: 30 Train set: Average loss: 0.5058, Accuracy: 82.216%, lr:0
Epoch: 30 Test set: Average loss: 0.6812, Accuracy: 76.920%
loss=0.5185818672180176 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 31 Train set: Average loss: 0.4838, Accuracy: 83.088%, lr:0
Epoch: 31 Test set: Average loss: 0.6480, Accuracy: 77.800%
loss=0.42589613795280457 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 32 Train set: Average loss: 0.4774, Accuracy: 83.272%, lr:0
Epoch: 32 Test set: Average loss: 0.5994, Accuracy: 79.500%
Model saved as Test Accuracy increased from 78.37 to 79.5 at epoch 32
loss=0.4766063690185547 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 33 Train set: Average loss: 0.4434, Accuracy: 84.338%, lr:0
Epoch: 33 Test set: Average loss: 0.6655, Accuracy: 77.680%
loss=0.4070991277694702 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 34 Train set: Average loss: 0.4287, Accuracy: 84.996%, lr:0
Epoch: 34 Test set: Average loss: 0.6244, Accuracy: 78.860%
loss=0.41572365164756775 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 35 Train set: Average loss: 0.4223, Accuracy: 85.228%, lr:0
Epoch: 35 Test set: Average loss: 0.6184, Accuracy: 78.810%
Epoch    35: reducing learning rate of group 0 to 2.0000e-02.
loss=0.31328195333480835 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 36 Train set: Average loss: 0.3303, Accuracy: 88.580%, lr:0
Epoch: 36 Test set: Average loss: 0.5387, Accuracy: 82.320%
Model saved as Test Accuracy increased from 79.5 to 82.32 at epoch 36
loss=0.29169192910194397 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 37 Train set: Average loss: 0.3027, Accuracy: 89.454%, lr:0
Epoch: 37 Test set: Average loss: 0.5382, Accuracy: 82.210%
loss=0.2520748972892761 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 38 Train set: Average loss: 0.2917, Accuracy: 89.982%, lr:0
Epoch: 38 Test set: Average loss: 0.5424, Accuracy: 81.970%
loss=0.28954026103019714 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 39 Train set: Average loss: 0.2808, Accuracy: 90.372%, lr:0
Epoch: 39 Test set: Average loss: 0.5297, Accuracy: 82.540%
Model saved as Test Accuracy increased from 82.32 to 82.54 at epoch 39
loss=0.2923586070537567 batch_id=97: 100% 98/98 [00:34<00:00,  2.87it/s]

Epoch: 40 Train set: Average loss: 0.2680, Accuracy: 90.880%, lr:0
Epoch: 40 Test set: Average loss: 0.5227, Accuracy: 82.990%
Model saved as Test Accuracy increased from 82.54 to 82.99 at epoch 40
Final model save path: /content/model_saves/model-32ec0ba3a6.pt  best Accuracy: 82.99
```
## Links
1. Model and Training code is packaged in **[this repo](https://github.com/rajy4683/mini-Rekog)**
2. Notebook used for training can be found **[here](https://github.com/rajy4683/EVA6/blob/master/EVA6S8/EVA6S8_SingleScript.ipynb)**

