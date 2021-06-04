# Assignment 6: MNIST Model Development in Stages



## Attempt 1

### Target:

1. Create a proper skeletal structure based on Receptive field and using: 

​    a. 3 conv Blocks with each block having 2 conv layers
​    b. Dropout and RELU after every convolution layer.
​    b. Final GAP Layer
​    c. Conv Block after GAP layer for prediction

2. Target accuracy of 99.4 in at least 1-2 runs

### Results: 
Total Parameters = 9034

```
Epoch: 8 Test set: Average loss: 0.0222, Accuracy: 9942/10000 (99.420%)
Epoch: 11 Test set: Average loss: 0.0209, Accuracy: 9942/10000 (99.420%)
Epoch: 13 Test set: Average loss: 0.0199, Accuracy: 9942/10000 (99.420%)
Epoch: 14 Test set: Average loss: 0.0230, Accuracy: 9940/10000 (99.400%)
```
### Analysis:

1. For 15 epochs consistent results achieved with tweaked LR and batch size = 64
2. Above 10th Epoch consistent accuracy was achieved
3. Used Padding in initial conv_blocks to prevent rapid image size reduction.
4. Two options for next model:
    a. Use deeper model to achieve higher accuracy earlier.
    b. Smaller and more compact model with even lesser params.
5. Given that already the model is able to perform well, the idea was to push the parameters down further and explore Augmentation/StepLR strategies

### Link

https://github.com/rajy4683/EVA6/blob/master/EVA6S5/S5EVA6_Attempt1.ipynb

## Attempt 2
### Target:
1. Reduce parameters from previous model by decreasing the number of channels in 2nd conv block
2. Introducing 1x1 between 1st conv block and 2nd conv block.
3. Augment images with Image Rotation of +/-7.0

### Results:
Total Parameters: 7310

    Epoch: 14 Test set: Average loss: 0.0182, Accuracy: 9943/10000 (99.430%)
    Epoch: 15 Test set: Average loss: 0.0198, Accuracy: 9940/10000 (99.400%)

### Link:
https://github.com/rajy4683/EVA6/blob/master/EVA6S5/S5EVA6_Attempt2.ipynb

## Attempt 3
### Analysis:
1. Base model performance achieve
2. Model is still in the underfitting zone
3. In this attempt training was done with and without Augmentation and observed atleast 0.5% jump in validation accuracy with Augmentation.
3. Last few epochs hover between 99.36-99.39.
4. The 1x1 looked a bit redundant as the channel count was same between the layers

Target:
1. Remove 1x1 layer in the 1st conv block
2. Try to retain accuracy of 99.4 by tuning LR/Batch size(128).
3. Retain image augmentation from the previous run.

Results: 

Total Parameters = 7288

    Epoch: 9 Test set: Average loss: 0.0204, Accuracy: 9939/10000 (99.390%)
    Epoch: 14 Test set: Average loss: 0.0187, Accuracy: 9940/10000 (99.400%)


Analysis:
1. The training loss stagnates over the run probably due to constant learning rate.
2. Still in underfitting zone
3. Doesn't achieve constant accuracy over multiple epochs

### Link:
https://github.com/rajy4683/EVA6/blob/master/EVA6S5/S5EVA6_Attempt3.ipynb

## Attempt 4 - Final
Target:
1. Achieve stable accuracy in last few 99.4
2. Add LR Scheduling
3. Tweak Dropout as the previous models were still in the underfitting zone.

Results: 

Total Parameters = 7288

    Epoch: 9 Test set: Average loss: 0.0204, Accuracy: 9939/10000 (99.390%)
    Epoch: 12 Test set: Average loss: 0.0196, Accuracy: 9940/10000 (99.400%)
    Epoch: 13 Test set: Average loss: 0.0194, Accuracy: 9940/10000 (99.400%)
    Epoch: 14 Test set: Average loss: 0.0193, Accuracy: 9942/10000 (99.420%)
    Epoch: 15 Test set: Average loss: 0.0194, Accuracy: 9941/10000 (99.410%)


Analysis:
1. Stable accuracy reached with both Image Augmentation and LR scheduling.
2. With LR scheduling, few observations were found:    
    a. If LR was reduced every run with a large gamma(0.1), then accuracy targets were not achieved    
    b. Till 8th epoch, model seemed to be happy with the default LR    
    c. Reducing LR (with gamma=0.09) post 8th epoch gave much more consistent results.

### Link:
https://github.com/rajy4683/EVA6/blob/master/EVA6S5/S5EVA6_FinalModel.ipynb