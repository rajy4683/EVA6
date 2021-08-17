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

![DETR Model](https://github.com/rajy4683/EVA6/blob/master/EVA6S14/imgs/detr_model.jpg)

![DETR Model](https://github.com/rajy4683/EVA6/blob/master/EVA6S14/imgs/detr_model2.jpg)

## Fine Tuning with DETR

Please refer to this **[notebook](https://github.com/rajy4683/EVA6/blob/master/EVA6S14/EVA6_S14DETR.ipynb)** or the  **[Colab Link](https://colab.research.google.com/drive/1QBbbFo-ICIzW_i6K0W1uN5117XxpIjkZ?usp=sharing)**  for fine-tuning on custom dataset.

Training logs and accuracy results are as below:

```
Averaged stats: class_error: 0.00  loss: 4.0744 (6.3327)  loss_ce: 0.0673 (0.2750)  loss_bbox: 0.1384 (0.1716)  loss_giou: 0.4137 (0.5738)  loss_ce_0: 0.2115 (0.3904)  loss_bbox_0: 0.1748 (0.2293)  loss_giou_0: 0.3938 (0.5671)  loss_ce_1: 0.1220 (0.2737)  loss_bbox_1: 0.1647 (0.2064)  loss_giou_1: 0.4214 (0.5702)  loss_ce_2: 0.1220 (0.2764)  loss_bbox_2: 0.1682 (0.2177)  loss_giou_2: 0.4583 (0.5819)  loss_ce_3: 0.0742 (0.2621)  loss_bbox_3: 0.1599 (0.1941)  loss_giou_3: 0.4489 (0.5587)  loss_ce_4: 0.0672 (0.2654)  loss_bbox_4: 0.1362 (0.1679)  loss_giou_4: 0.4146 (0.5510)  loss_ce_unscaled: 0.0673 (0.2750)  class_error_unscaled: 0.0000 (13.2035)  loss_bbox_unscaled: 0.0277 (0.0343)  loss_giou_unscaled: 0.2069 (0.2869)  cardinality_error_unscaled: 1.5000 (1.6429)  loss_ce_0_unscaled: 0.2115 (0.3904)  loss_bbox_0_unscaled: 0.0350 (0.0459)  loss_giou_0_unscaled: 0.1969 (0.2836)  cardinality_error_0_unscaled: 7.0000 (6.1429)  loss_ce_1_unscaled: 0.1220 (0.2737)  loss_bbox_1_unscaled: 0.0329 (0.0413)  loss_giou_1_unscaled: 0.2107 (0.2851)  cardinality_error_1_unscaled: 3.5000 (3.6429)  loss_ce_2_unscaled: 0.1220 (0.2764)  loss_bbox_2_unscaled: 0.0336 (0.0435)  loss_giou_2_unscaled: 0.2292 (0.2909)  cardinality_error_2_unscaled: 1.5000 (2.8571)  loss_ce_3_unscaled: 0.0742 (0.2621)  loss_bbox_3_unscaled: 0.0320 (0.0388)  loss_giou_3_unscaled: 0.2245 (0.2794)  cardinality_error_3_unscaled: 1.0000 (1.8571)  loss_ce_4_unscaled: 0.0672 (0.2654)  loss_bbox_4_unscaled: 0.0272 (0.0336)  loss_giou_4_unscaled: 0.2073 (0.2755)  cardinality_error_4_unscaled: 1.0000 (1.5714)
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.735
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.701
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.770
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
```

![Accuracy plots](https://github.com/rajy4683/EVA6/blob/master/EVA6S14/imgs/traininglogs.jpg))



## References:

1. [DETR Paper](https://arxiv.org/abs/2005.12872)
2. [Article on Finetuning DETR](https://opensourcelibs.com/lib/finetune-detr)

