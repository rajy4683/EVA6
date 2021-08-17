# DETR - Detection Transformer
DETR is Transformer -based model that can perform end-to-end object detection. The standard DETR architecture combines the following :

1. A Convolutional Backbone that creates necessary features.
2. A Transformer network with both Encoder-Decoder networks that have 2 prediction heads corresponding to classes and bounding boxes
3. Set-based global loss function.

Compared to Vision Transformers(ViT), DETR targets end-to-end object detection and also uses an Encoder-Decoder architecture, while ViT is an Encoder only architecture. Additionally, instead of directly splitting the image into predefined patches, DETR uses a CNN backbone(ResNet50) to both scale the image and create rich features. E.g: A standard HxWx3 is converted into (H/32 x W/32 x 2048).These features/channels are further scaled down with 1x1 convolution to H/32 x W/32 x 256. The backbone's output is fed into the Self-attention and MLP layers of a standard Transformer. 

The decoder side of the DETR has major modifications compared to standard decoder in Transformer layers. A standard Transformer layer(in NLP context) is usually auto-regressive in nature i.e single output token is predicted and used for further prediction. However, in DETR architecture, parallel decoding/prediction is performed by feeding multiple(100) randomly initialized and learnable parameters to the decoder. These weights are called *object queries* and are positional encodings that also form the input(i.e Q vector) to both the Self-Attention layer and the Encoder-Decoder Attention Layer.  

The final output layer of DETR consists of two blocks/heads one for predicting class of the object and other for predicting bounding boxes. Each of these heads can predict maximum of 100 classes and bounding boxes. Since DETR predicts a fixed-size set of 100 bounding boxes, which may be much larger than the actual number of objects of interest in an image, an additional special class label ∅(`no_object`) is used to represent that no object is detected within a slot. This class plays a similar role to the “background” class in the standard object detection approaches. The input/ground truth images will usually consist of much lesser number of bounding boxes/classes. Hence, to match the dimensions during loss calculation, the ground truth is padded with the special `no_object` class for all empty slots. 

Now the comparison between ground truth and the predictions becomes a set prediction problem where the objective is to identify closest matching entities in ground truth and DETR prediction. To obtain the most optimal value of such a matrix, **[Hungarian Matching Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)** is utilized. This algorithm uses a bipartite graph to derive the best matching pairs. The cost of each matches are based on two types of losses: 

1. For class predictions: Cross Entropy loss  
2. For bounding-box losses: Generalized Intersection-Over-Union  + L1 Norm.

![DETR Model]()

![DETR Model]()

## Fine Tuning with DETR

Please refer to this **[notebook](https://github.com/rajy4683/EVA6/blob/master/EVA6S12/EVA6_S12_STN.ipynb)** or the  **[Colab Link](https://colab.research.google.com/drive/1eyJ7F6tvvRjh9uu8kUzi67CB2HMHHHnC?usp=sharing)**  for fine-tuning on custom dataset.



## References:

1. [DETR Paper](https://arxiv.org/abs/2005.12872)
2. [Article on Finetuning DETR](https://opensourcelibs.com/lib/finetune-detr)

