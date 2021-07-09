# Custom ResNet Models on CIFAR10

In this repo, we evaluate modified ResNet18 model (using LayerNorm instead of BatchNormalization) to achieve **91.63%** test accuracy on CIFAR-10 dataset. 

Notebook used for training can be found [here](https://github.com/rajy4683/EVA6/blob/master/EVA6S9/EVA6S9_SingleScript.ipynb)

The main repo used for training the models can be found [here](https://github.com/rajy4683/mini-Rekog)

## Model Summary

The final model has 4 Convolution blocks with 6,573,130 parameters.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
 ModifiedResBlock-14          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         294,912
        MaxPool2d-16            [-1, 256, 8, 8]               0
      BatchNorm2d-17            [-1, 256, 8, 8]             512
             ReLU-18            [-1, 256, 8, 8]               0
           Conv2d-19            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-20            [-1, 512, 4, 4]               0
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
             ReLU-22            [-1, 512, 4, 4]               0
           Conv2d-23            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
             ReLU-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
             ReLU-28            [-1, 512, 4, 4]               0
 ModifiedResBlock-29            [-1, 512, 4, 4]               0
        MaxPool2d-30            [-1, 512, 1, 1]               0
           Linear-31                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.75
Params size (MB): 25.07
Estimated Total Size (MB): 31.84
----------------------------------------------------------------
```
## Image Augmentation
[Albumentation Library](https://github.com/albumentations-team/albumentations) was primarily used for augmentation during training. Following schemes were chosen:
1. RandomCrop and Pad
2. HorizontalFlip
3. CutOut(8x8)

Samples of Augmented images are below:

![Augmented Images](https://github.com/rajy4683/EVA6/blob/master/EVA6S9/imgs/EVA6S9_SampleAugmentation.png)

## Learning Rate Range Test
For training custom models with Cyclic LR policies is an extremely good option. To perform a proper assessment of Learning rates LRFinder was used. Below are the results from LR Range test. 
![LR Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S9/imgs/EVA6S9_LRTest.png)

LRFinder code was used from this [repo](https://github.com/davidtvs/pytorch-lr-finder)

However, for best results, Maximum LR used was 0.1 with OneCycleLRPolicy. 

## Training Logs

Training was performed with using ReduceLROnPlateau for 40 epochs. Below are the accuracy and loss plots:
![Accuracy Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S9/imgs/EVA6S9_Accuracy.png)
![Loss Plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S9/imgs/EVA6S9_LossPlot.png)

```
loss=1.5639212131500244 batch_id=97: 100% 98/98 [00:15<00:00,  6.41it/s]

Epoch: 1 Train set: Average loss: 1.9113, Accuracy: 37.370%, lr:0.020385296927999084
Epoch: 1 Test set: Average loss: 1.5512, Accuracy: 50.710%
Model saved as Test Accuracy increased from 0.0 to 50.71 at epoch 1
loss=1.1206406354904175 batch_id=97: 100% 98/98 [00:15<00:00,  6.47it/s]

Epoch: 2 Train set: Average loss: 1.6300, Accuracy: 51.252%, lr:0.04037996885599817
Epoch: 2 Test set: Average loss: 1.2646, Accuracy: 58.360%
Model saved as Test Accuracy increased from 50.71 to 58.36 at epoch 2
loss=1.086284875869751 batch_id=97: 100% 98/98 [00:15<00:00,  6.37it/s]

Epoch: 3 Train set: Average loss: 1.2058, Accuracy: 62.860%, lr:0.060374640783997256
Epoch: 3 Test set: Average loss: 1.6050, Accuracy: 63.410%
Model saved as Test Accuracy increased from 58.36 to 63.41 at epoch 3
loss=1.1450005769729614 batch_id=97: 100% 98/98 [00:15<00:00,  6.40it/s]

Epoch: 4 Train set: Average loss: 1.0190, Accuracy: 69.316%, lr:0.08036931271199635
Epoch: 4 Test set: Average loss: 1.3197, Accuracy: 63.790%
Model saved as Test Accuracy increased from 63.41 to 63.79 at epoch 4
loss=0.7415210008621216 batch_id=97: 100% 98/98 [00:15<00:00,  6.34it/s]

Epoch: 5 Train set: Average loss: 1.0457, Accuracy: 70.046%, lr:0.09990423229120281
Epoch: 5 Test set: Average loss: 0.8421, Accuracy: 73.300%
Model saved as Test Accuracy increased from 63.79 to 73.3 at epoch 5
loss=0.5880386829376221 batch_id=97: 100% 98/98 [00:15<00:00,  6.33it/s]

Epoch: 6 Train set: Average loss: 0.7440, Accuracy: 77.054%, lr:0.09464345008149178
Epoch: 6 Test set: Average loss: 0.6851, Accuracy: 78.860%
Model saved as Test Accuracy increased from 73.3 to 78.86 at epoch 6
loss=0.6777477264404297 batch_id=97: 100% 98/98 [00:15<00:00,  6.34it/s]

Epoch: 7 Train set: Average loss: 0.6002, Accuracy: 80.414%, lr:0.08938266787178073
Epoch: 7 Test set: Average loss: 0.6398, Accuracy: 80.940%
Model saved as Test Accuracy increased from 78.86 to 80.94 at epoch 7
loss=0.3876872658729553 batch_id=97: 100% 98/98 [00:15<00:00,  6.34it/s]

Epoch: 8 Train set: Average loss: 0.5064, Accuracy: 83.252%, lr:0.08412188566206968
Epoch: 8 Test set: Average loss: 0.4493, Accuracy: 85.110%
Model saved as Test Accuracy increased from 80.94 to 85.11 at epoch 8
loss=0.46819356083869934 batch_id=97: 100% 98/98 [00:15<00:00,  6.38it/s]

Epoch: 9 Train set: Average loss: 0.4131, Accuracy: 85.790%, lr:0.07886110345235864
Epoch: 9 Test set: Average loss: 0.4628, Accuracy: 84.950%
loss=0.2762899696826935 batch_id=97: 100% 98/98 [00:15<00:00,  6.42it/s]

Epoch: 10 Train set: Average loss: 0.3893, Accuracy: 86.600%, lr:0.07360032124264759
Epoch: 10 Test set: Average loss: 0.4157, Accuracy: 85.920%
Model saved as Test Accuracy increased from 85.11 to 85.92 at epoch 10
loss=0.2862881124019623 batch_id=97: 100% 98/98 [00:15<00:00,  6.40it/s]

Epoch: 11 Train set: Average loss: 0.3427, Accuracy: 88.090%, lr:0.06833953903293655
Epoch: 11 Test set: Average loss: 0.3746, Accuracy: 87.610%
Model saved as Test Accuracy increased from 85.92 to 87.61 at epoch 11
loss=0.3531803488731384 batch_id=97: 100% 98/98 [00:15<00:00,  6.41it/s]

Epoch: 12 Train set: Average loss: 0.3088, Accuracy: 89.252%, lr:0.06307875682322552
Epoch: 12 Test set: Average loss: 0.4020, Accuracy: 87.290%
loss=0.294404000043869 batch_id=97: 100% 98/98 [00:15<00:00,  6.38it/s]

Epoch: 13 Train set: Average loss: 0.2855, Accuracy: 90.060%, lr:0.05781797461351446
Epoch: 13 Test set: Average loss: 0.4125, Accuracy: 86.440%
loss=0.2793017327785492 batch_id=97: 100% 98/98 [00:15<00:00,  6.37it/s]

Epoch: 14 Train set: Average loss: 0.2542, Accuracy: 91.118%, lr:0.052557192403803424
Epoch: 14 Test set: Average loss: 0.4323, Accuracy: 86.470%
loss=0.23361358046531677 batch_id=97: 100% 98/98 [00:15<00:00,  6.40it/s]

Epoch: 15 Train set: Average loss: 0.2252, Accuracy: 92.190%, lr:0.04729641019409238
Epoch: 15 Test set: Average loss: 0.3328, Accuracy: 89.360%
Model saved as Test Accuracy increased from 87.61 to 89.36 at epoch 15
loss=0.20086196064949036 batch_id=97: 100% 98/98 [00:15<00:00,  6.42it/s]

Epoch: 16 Train set: Average loss: 0.2054, Accuracy: 92.730%, lr:0.04203562798438134
Epoch: 16 Test set: Average loss: 0.3310, Accuracy: 89.560%
Model saved as Test Accuracy increased from 89.36 to 89.56 at epoch 16
loss=0.22469854354858398 batch_id=97: 100% 98/98 [00:15<00:00,  6.39it/s]

Epoch: 17 Train set: Average loss: 0.1836, Accuracy: 93.488%, lr:0.0367748457746703
Epoch: 17 Test set: Average loss: 0.3350, Accuracy: 89.670%
Model saved as Test Accuracy increased from 89.56 to 89.67 at epoch 17
loss=0.1494419276714325 batch_id=97: 100% 98/98 [00:15<00:00,  6.38it/s]

Epoch: 18 Train set: Average loss: 0.1720, Accuracy: 93.944%, lr:0.031514063564959255
Epoch: 18 Test set: Average loss: 0.3807, Accuracy: 88.480%
loss=0.15254512429237366 batch_id=97: 100% 98/98 [00:15<00:00,  6.43it/s]

Epoch: 19 Train set: Average loss: 0.1527, Accuracy: 94.712%, lr:0.02625328135524821
Epoch: 19 Test set: Average loss: 0.3247, Accuracy: 89.880%
Model saved as Test Accuracy increased from 89.67 to 89.88 at epoch 19
loss=0.10115960985422134 batch_id=97: 100% 98/98 [00:15<00:00,  6.44it/s]

Epoch: 20 Train set: Average loss: 0.1337, Accuracy: 95.304%, lr:0.020992499145537177
Epoch: 20 Test set: Average loss: 0.3072, Accuracy: 91.000%
Model saved as Test Accuracy increased from 89.88 to 91.0 at epoch 20
loss=0.1427859514951706 batch_id=97: 100% 98/98 [00:15<00:00,  6.33it/s]

Epoch: 21 Train set: Average loss: 0.1256, Accuracy: 95.576%, lr:0.015731716935826118
Epoch: 21 Test set: Average loss: 0.2918, Accuracy: 91.090%
Model saved as Test Accuracy increased from 91.0 to 91.09 at epoch 21
loss=0.10478507727384567 batch_id=97: 100% 98/98 [00:15<00:00,  6.32it/s]

Epoch: 22 Train set: Average loss: 0.1059, Accuracy: 96.278%, lr:0.010470934726115086
Epoch: 22 Test set: Average loss: 0.2923, Accuracy: 91.170%
Model saved as Test Accuracy increased from 91.09 to 91.17 at epoch 22
loss=0.09049015492200851 batch_id=97: 100% 98/98 [00:15<00:00,  6.25it/s]

Epoch: 23 Train set: Average loss: 0.0939, Accuracy: 96.778%, lr:0.00521015251640404
Epoch: 23 Test set: Average loss: 0.3027, Accuracy: 91.440%
Model saved as Test Accuracy increased from 91.17 to 91.44 at epoch 23
loss=0.1021457388997078 batch_id=97: 100% 98/98 [00:15<00:00,  6.34it/s]

Epoch: 24 Train set: Average loss: 0.0853, Accuracy: 97.022%, lr:-5.062969330700551e-05
Epoch: 24 Test set: Average loss: 0.2928, Accuracy: 91.630%
Model saved as Test Accuracy increased from 91.44 to 91.63 at epoch 24
Final model save path: /content/model_saves/model-2c1e69cefa.pt  best Accuracy: 91.63
```
## Links
1. Model and Training code is packaged in **[this repo](https://github.com/rajy4683/mini-Rekog)**
2. Notebook used for training can be found **[here](https://github.com/rajy4683/EVA6/blob/master/EVA6S9/EVA6S9_SingleScript.ipynb)**

## References
1. https://github.com/bckenstler/CLR
2. https://github.com/davidtvs/pytorch-lr-finder