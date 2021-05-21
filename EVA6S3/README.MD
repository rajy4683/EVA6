# EVA6 S3 - MNIST Adder

In this project we will be creating a neural network model that can do the following:

### Inputs:

1.  Accept an image from the MNIST dataset and
2. A random number between 0 and 9

### Outputs:

1. The class label represented by the MNIST image
2. Sum of the random number and the MNIST image.

### Data Representation:

1. Each image from MNIST test/training dataset is represented as a Tensor. Both the test and training images are normalized by the MNIST dataset mean and variance.  
2. The labels are represented as the decimal number between 0 and 9
3. The random number(between 0-9) to be added is represented as a one-hot encoded array of 10.
4. The sum of the random number and the class label as a decimal number (0-18)

### Data Generation

1. A simple generator class for both is used for both training and test dataset which generates a 4-tuple:
   1. Tensor of the image of dimension (1x28x28)
   2. Tensor of Random number as one-hot encoded 
   3. Target/Label of the image as an Integer 
   4. The sum Target and Random number as an integer
2. torch.dataloaders are created for both training and test data with:
   1. Batch size = 128
   2. Shuffle = True
   3. Memory pinning enabled.

### NN Model

![Model Definition](https://github.com/rajy4683/EVA6/blob/master/EVA6S3/basemodel.png)
![Traced Model](https://github.com/rajy4683/EVA6/blob/master/EVA6S3/traced.png)

### Loss Function:

Total Loss = NLL loss on classification + NLL loss on Addition.

The core loss function was PyTorch's F.Nllloss function. This is because the entire problem is posed as a classification task and the final output layer also uses a LogSoftmax which in turn outputs log-probabilities across classes. For the image classification, the classes are between 0-9 while for addition the classes are between 0-18.

### Results and Evaluation:

Results were evaluated using the predicted vs actual accuracy on the Test Dataset for both classification and addition tasks:

Below is the sample output:

```
Epoch: 47 Test set: Average class loss: 0.02350032809365621, Accuracy: 9964/10000 (99.64%)
Epoch: 47 Test set: Average Sum Loss:0.23227927067875861 Accuracy: 9861/10000 (98.61%)
Epoch: 48 Test set: *Average class loss*: 0.02483714062999611, Accuracy: 9965/10000 (99.65%)
Epoch: 48 Test set: *Average Sum Loss*:0.2246130622267723 Accuracy: 9858/10000 (98.58%)
Epoch: 49 Test set: Average class loss: 0.025573136833245372, Accuracy: 9961/10000 (99.61%)
```

### Training Logs:

`

```
Epoch: 1 Test set: Average class loss: 0.029781136710895224, Accuracy: 9909/10000 (99.09%)
Epoch: 1 Test set: Average Sum Loss:2.1067971336364746 Accuracy: 2077/10000 (20.77%)
Epoch: 2 Test set: Average class loss: 0.02811167477183044, Accuracy: 9922/10000 (99.22%)
Epoch: 2 Test set: Average Sum Loss:1.6257107707977294 Accuracy: 3710/10000 (37.1%)
Epoch: 3 Test set: Average class loss: 0.021551052032783628, Accuracy: 9935/10000 (99.35%)
Epoch: 3 Test set: Average Sum Loss:1.3302183277130126 Accuracy: 4979/10000 (49.79%)
Epoch: 4 Test set: Average class loss: 0.020353321626968682, Accuracy: 9938/10000 (99.38%)
Epoch: 4 Test set: Average Sum Loss:1.1584116191864013 Accuracy: 5411/10000 (54.11%)
Epoch: 5 Test set: Average class loss: 0.023299573473073543, Accuracy: 9940/10000 (99.4%)
Epoch: 5 Test set: Average Sum Loss:1.0022976754188537 Accuracy: 6771/10000 (67.71%)
Epoch: 6 Test set: Average class loss: 0.02777718466236256, Accuracy: 9931/10000 (99.31%)
Epoch: 6 Test set: Average Sum Loss:0.9111501329421997 Accuracy: 7214/10000 (72.14%)
Epoch: 7 Test set: Average class loss: 0.021386632597073914, Accuracy: 9943/10000 (99.43%)
Epoch: 7 Test set: Average Sum Loss:0.8184787579536438 Accuracy: 7434/10000 (74.34%)
Epoch: 8 Test set: Average class loss: 0.023849810665560654, Accuracy: 9945/10000 (99.45%)
Epoch: 8 Test set: Average Sum Loss:0.7537479559898377 Accuracy: 7798/10000 (77.98%)
Epoch: 9 Test set: Average class loss: 0.02283721186830662, Accuracy: 9944/10000 (99.44%)
Epoch: 9 Test set: Average Sum Loss:0.685794631767273 Accuracy: 8613/10000 (86.13%)
Epoch: 10 Test set: Average class loss: 0.02070178262074478, Accuracy: 9950/10000 (99.5%)
Epoch: 10 Test set: Average Sum Loss:0.6359557472229004 Accuracy: 8695/10000 (86.95%)
Epoch: 11 Test set: Average class loss: 0.033498095539142375, Accuracy: 9938/10000 (99.38%)
Epoch: 11 Test set: Average Sum Loss:0.6637102916717529 Accuracy: 8035/10000 (80.35%)
Epoch: 12 Test set: Average class loss: 0.023731720602923452, Accuracy: 9949/10000 (99.49%)
Epoch: 12 Test set: Average Sum Loss:0.5852170094013214 Accuracy: 8882/10000 (88.82%)
Epoch: 13 Test set: Average class loss: 0.05262757916972041, Accuracy: 9870/10000 (98.7%)
Epoch: 13 Test set: Average Sum Loss:0.6625428275108337 Accuracy: 8539/10000 (85.39%)
Epoch: 14 Test set: Average class loss: 0.02647817108957097, Accuracy: 9942/10000 (99.42%)
Epoch: 14 Test set: Average Sum Loss:0.5305684957504272 Accuracy: 8964/10000 (89.64%)
Epoch: 15 Test set: Average class loss: 0.017459278390795224, Accuracy: 9955/10000 (99.55%)
Epoch: 15 Test set: Average Sum Loss:0.46564145975112914 Accuracy: 9336/10000 (93.36%)
Epoch: 16 Test set: Average class loss: 0.026828678897459757, Accuracy: 9940/10000 (99.4%)
Epoch: 16 Test set: Average Sum Loss:0.5220946075439453 Accuracy: 8944/10000 (89.44%)
Epoch: 17 Test set: Average class loss: 0.023511913765931966, Accuracy: 9953/10000 (99.53%)
Epoch: 17 Test set: Average Sum Loss:0.4216055794715881 Accuracy: 9471/10000 (94.71%)
Epoch: 18 Test set: Average class loss: 0.06339229862214997, Accuracy: 9857/10000 (98.57%)
Epoch: 18 Test set: Average Sum Loss:0.6688988928794861 Accuracy: 9036/10000 (90.36%)
Epoch: 19 Test set: Average class loss: 0.03318554861416633, Accuracy: 9941/10000 (99.41%)
Epoch: 19 Test set: Average Sum Loss:0.43094079465866086 Accuracy: 9617/10000 (96.17%)
Epoch: 20 Test set: Average class loss: 0.04862264170163544, Accuracy: 9930/10000 (99.3%)
Epoch: 20 Test set: Average Sum Loss:0.40607184953689573 Accuracy: 9566/10000 (95.66%)
Epoch: 21 Test set: Average class loss: 0.0254979210500911, Accuracy: 9950/10000 (99.5%)
Epoch: 21 Test set: Average Sum Loss:0.35637129957675934 Accuracy: 9788/10000 (97.88%)
Epoch: 22 Test set: Average class loss: 0.025529943281190935, Accuracy: 9949/10000 (99.49%)
Epoch: 22 Test set: Average Sum Loss:0.3438975245475769 Accuracy: 9734/10000 (97.34%)
Epoch: 23 Test set: Average class loss: 0.02189993600859307, Accuracy: 9956/10000 (99.56%)
Epoch: 23 Test set: Average Sum Loss:0.3285067180633545 Accuracy: 9766/10000 (97.66%)
Epoch: 24 Test set: Average class loss: 0.023458078561956062, Accuracy: 9958/10000 (99.58%)
Epoch: 24 Test set: Average Sum Loss:0.30426155021190643 Accuracy: 9836/10000 (98.36%)
Epoch: 25 Test set: Average class loss: 0.020542383010005143, Accuracy: 9962/10000 (99.62%)
Epoch: 25 Test set: Average Sum Loss:0.2821488481283188 Accuracy: 9816/10000 (98.16%)
Epoch: 26 Test set: Average class loss: 0.02290178811738151, Accuracy: 9965/10000 (99.65%)
Epoch: 26 Test set: Average Sum Loss:0.28253089673519133 Accuracy: 9837/10000 (98.37%)
Epoch: 27 Test set: Average class loss: 0.0242400248936945, Accuracy: 9957/10000 (99.57%)
Epoch: 27 Test set: Average Sum Loss:0.2910739560723305 Accuracy: 9822/10000 (98.22%)
Epoch: 28 Test set: Average class loss: 0.0254696618508955, Accuracy: 9963/10000 (99.63%)
Epoch: 28 Test set: Average Sum Loss:0.28359835546016693 Accuracy: 9834/10000 (98.34%)
Epoch: 29 Test set: Average class loss: 0.031546996685347405, Accuracy: 9945/10000 (99.45%)
Epoch: 29 Test set: Average Sum Loss:0.38362774262428284 Accuracy: 9716/10000 (97.16%)
Epoch: 30 Test set: Average class loss: 0.045796580664697106, Accuracy: 9928/10000 (99.28%)
Epoch: 30 Test set: Average Sum Loss:0.4440179497361183 Accuracy: 9626/10000 (96.26%)
Epoch: 31 Test set: Average class loss: 0.03860998377521755, Accuracy: 9943/10000 (99.43%)
Epoch: 31 Test set: Average Sum Loss:0.3601237511396408 Accuracy: 9710/10000 (97.1%)
Epoch: 32 Test set: Average class loss: 0.033688783714734015, Accuracy: 9942/10000 (99.42%)
Epoch: 32 Test set: Average Sum Loss:0.3172424297332764 Accuracy: 9769/10000 (97.69%)
Epoch: 33 Test set: Average class loss: 0.026887675154971657, Accuracy: 9957/10000 (99.57%)
Epoch: 33 Test set: Average Sum Loss:0.25456808080673216 Accuracy: 9861/10000 (98.61%)
Epoch: 34 Test set: Average class loss: 0.024305231332418045, Accuracy: 9959/10000 (99.59%)
Epoch: 34 Test set: Average Sum Loss:0.23246212162971497 Accuracy: 9870/10000 (98.7%)
Epoch: 35 Test set: Average class loss: 0.022887673134137004, Accuracy: 9958/10000 (99.58%)
Epoch: 35 Test set: Average Sum Loss:0.2597497302055359 Accuracy: 9858/10000 (98.58%)
Epoch: 36 Test set: Average class loss: 0.023152222563140094, Accuracy: 9962/10000 (99.62%)
Epoch: 36 Test set: Average Sum Loss:0.248951704287529 Accuracy: 9862/10000 (98.62%)
Epoch: 37 Test set: Average class loss: 0.025881195527051204, Accuracy: 9956/10000 (99.56%)
Epoch: 37 Test set: Average Sum Loss:0.27403179619312285 Accuracy: 9849/10000 (98.49%)
Epoch: 38 Test set: Average class loss: 0.022820679961220593, Accuracy: 9962/10000 (99.62%)
Epoch: 38 Test set: Average Sum Loss:0.24652360544204713 Accuracy: 9874/10000 (98.74%)
Epoch: 39 Test set: Average class loss: 0.02632438176128744, Accuracy: 9953/10000 (99.53%)
Epoch: 39 Test set: Average Sum Loss:0.2742140605092049 Accuracy: 9810/10000 (98.1%)
Epoch: 40 Test set: Average class loss: 0.025465426477925213, Accuracy: 9964/10000 (99.64%)
Epoch: 40 Test set: Average Sum Loss:0.2462636006951332 Accuracy: 9861/10000 (98.61%)
Epoch: 41 Test set: Average class loss: 0.024458243751239207, Accuracy: 9963/10000 (99.63%)
Epoch: 41 Test set: Average Sum Loss:0.2562820773601532 Accuracy: 9860/10000 (98.6%)
Epoch: 42 Test set: Average class loss: 0.02505271040558173, Accuracy: 9962/10000 (99.62%)
Epoch: 42 Test set: Average Sum Loss:0.23173072811961173 Accuracy: 9863/10000 (98.63%)
Epoch: 43 Test set: Average class loss: 0.024856062382276285, Accuracy: 9959/10000 (99.59%)
Epoch: 43 Test set: Average Sum Loss:0.24460083354711532 Accuracy: 9868/10000 (98.68%)
Epoch: 44 Test set: Average class loss: 0.02416417478397143, Accuracy: 9962/10000 (99.62%)
Epoch: 44 Test set: Average Sum Loss:0.2557578363180161 Accuracy: 9863/10000 (98.63%)
Epoch: 45 Test set: Average class loss: 0.024762619278957208, Accuracy: 9961/10000 (99.61%)
Epoch: 45 Test set: Average Sum Loss:0.24993746840953826 Accuracy: 9867/10000 (98.67%)
Epoch: 46 Test set: Average class loss: 0.024542020087075435, Accuracy: 9964/10000 (99.64%)
Epoch: 46 Test set: Average Sum Loss:0.2646421900510788 Accuracy: 9848/10000 (98.48%)
Epoch: 47 Test set: Average class loss: 0.02350032809365621, Accuracy: 9964/10000 (99.64%)
Epoch: 47 Test set: Average Sum Loss:0.23227927067875861 Accuracy: 9861/10000 (98.61%)
Epoch: 48 Test set: Average class loss: 0.02483714062999611, Accuracy: 9965/10000 (99.65%)
Epoch: 48 Test set: Average Sum Loss:0.2246130622267723 Accuracy: 9858/10000 (98.58%)
Epoch: 49 Test set: Average class loss: 0.025573136833245372, Accuracy: 9961/10000 (99.61%)
Epoch: 49 Test set: Average Sum Loss:0.2610436644792557 Accuracy: 9852/10000 (98.52%)
```

Training logs`

