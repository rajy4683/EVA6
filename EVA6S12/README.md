# Spatial Transformer Networks
With Convolutional Neural networks, one of the continued challenge is to attain spatial invariance. Spatial transformers, introduced by [this paper](https://arxiv.org/pdf/1506.02025.pdf) provides a new learnable/differentiable module that handles spatial manipulation alongside CNN layers.   While the feature maps tend to learn edges, gradients, textures, patterns etc, Spatial Transformer blocks learn transformations such as scaling, cropping, rotation and also non-rigid body deformations. In effect, the Spatial transformers enable CNN networks to pay attention to spatial orientation in the images and also identify closely related features within a set of feature maps.

The Spatial Transformer Network consists of following components:

1. **Localization Network**: This network takes a set of Input feature maps and creates a transformation matrix of 3x2. 
2. **Parameterised Sampling Grid**: Using the output of the affine/transformation matrix learnt above, this layer samples points from the input feature map to produce the final output transformation.

Below is the architecture of the network:

![Spatial Transformer Network](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/imgs/STN.JPG)

In essence, an STN can be created using any combination of CNN and/or Linear layers, however, the output must be the transformation matrix that can be applied to the feature map.

## Model for CIFAR-10 Dataset

Please refer to this **[notebook](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/EVA6_S12_STN.ipynb)** or the  **[Colab Link](https://colab.research.google.com/drive/1eyJ7F6tvvRjh9uu8kUzi67CB2HMHHHnC?usp=sharing)**  for complete implementation and training mechanism.

We use a modified version of the STN which has 3 Convolutional Layers followed by Max Polling and Final linear blocks that outputs the final transformation matrix of shape [B, C, 3, 2]. Next we use PyTorch's `affine_grid` and `grid_sample` functions to combine the transformation matrix with the input feature image. The output of the STN block is finally passed through a Custom CNN network to perform final classification.

Below is the model summary:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 30, 30]             224
            Conv2d-2            [-1, 8, 28, 28]             584
            Conv2d-3            [-1, 8, 22, 22]           3,144
         MaxPool2d-4            [-1, 8, 11, 11]               0
              ReLU-5            [-1, 8, 11, 11]               0
            Conv2d-6             [-1, 10, 7, 7]           2,010
         MaxPool2d-7             [-1, 10, 3, 3]               0
              ReLU-8             [-1, 10, 3, 3]               0
            Linear-9                   [-1, 32]           2,912
             ReLU-10                   [-1, 32]               0
           Linear-11                    [-1, 6]             198
           Conv2d-12           [-1, 24, 32, 32]             648
      BatchNorm2d-13           [-1, 24, 32, 32]              48
             ReLU-14           [-1, 24, 32, 32]               0
          Dropout-15           [-1, 24, 32, 32]               0
           Conv2d-16           [-1, 24, 32, 32]           5,184
      BatchNorm2d-17           [-1, 24, 32, 32]              48
             ReLU-18           [-1, 24, 32, 32]               0
          Dropout-19           [-1, 24, 32, 32]               0
           Conv2d-20           [-1, 24, 16, 16]           5,184
      BatchNorm2d-21           [-1, 24, 16, 16]              48
             ReLU-22           [-1, 24, 16, 16]               0
          Dropout-23           [-1, 24, 16, 16]               0
           Conv2d-24           [-1, 24, 16, 16]             576
           Conv2d-25           [-1, 24, 16, 16]           5,184
      BatchNorm2d-26           [-1, 24, 16, 16]              48
             ReLU-27           [-1, 24, 16, 16]               0
          Dropout-28           [-1, 24, 16, 16]               0
           Conv2d-29           [-1, 24, 16, 16]             216
      BatchNorm2d-30           [-1, 24, 16, 16]              48
             ReLU-31           [-1, 24, 16, 16]               0
          Dropout-32           [-1, 24, 16, 16]               0
           Conv2d-33             [-1, 24, 8, 8]             432
      BatchNorm2d-34             [-1, 24, 8, 8]              48
             ReLU-35             [-1, 24, 8, 8]               0
          Dropout-36             [-1, 24, 8, 8]               0
           Conv2d-37             [-1, 48, 8, 8]           1,152
           Conv2d-38             [-1, 48, 8, 8]          20,736
      BatchNorm2d-39             [-1, 48, 8, 8]              96
             ReLU-40             [-1, 48, 8, 8]               0
          Dropout-41             [-1, 48, 8, 8]               0
           Conv2d-42             [-1, 48, 8, 8]             432
      BatchNorm2d-43             [-1, 48, 8, 8]              96
             ReLU-44             [-1, 48, 8, 8]               0
          Dropout-45             [-1, 48, 8, 8]               0
           Conv2d-46             [-1, 96, 8, 8]           4,608
           Conv2d-47             [-1, 96, 8, 8]          82,944
      BatchNorm2d-48             [-1, 96, 8, 8]             192
             ReLU-49             [-1, 96, 8, 8]               0
          Dropout-50             [-1, 96, 8, 8]               0
           Conv2d-51             [-1, 96, 8, 8]           6,912
      BatchNorm2d-52             [-1, 96, 8, 8]             192
             ReLU-53             [-1, 96, 8, 8]               0
          Dropout-54             [-1, 96, 8, 8]               0
           Conv2d-55             [-1, 10, 8, 8]             960
        AvgPool2d-56             [-1, 10, 1, 1]               0
CIFARModelDDilate-57                   [-1, 10]               0
================================================================
Total params: 145,104
Trainable params: 145,104
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.95
Params size (MB): 0.55
Estimated Total Size (MB): 3.51
----------------------------------------------------------------
```

## Results and Summary

We achieve **~78.9%** accuracy on CIFAR10 which is significantly lower compared to the base model performance. In this experiment, no augmentation was used. Below are some of the samples for learned transformations for the image samples.

![Stage 1](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/imgs/STN40.png)

![Stage 2](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/imgs/STN64.png)

![Stage 3](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/imgs/STN78.png)

![Accuracy Plot](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/imgs/EVA6S12_Accuracy.png)

## References:

1. [DeepMind Paper](https://arxiv.org/abs/1506.02025)
2. [PyTorch Tutorial on Spatial Transformer Networks](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html)
