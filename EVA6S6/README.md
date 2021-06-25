# Assignment 6: Testing multiple Normalization types with L1 and L2 Losses

## Objective
We study the impact of using Normlization techniques (Batch Normalization, Layer Normalization and Group Normalization) and Regularization techniques(L1 and L2 ) on model performance and convergence. We use the baseline model from [here](https://github.com/rajy4683/EVA6/blob/master/EVA6S5/S5EVA6_FinalModel.ipynb)

## Model definition:
The model definition can be found [here](https://github.com/rajy4683/mini-Rekog/blob/f10c8413fa0b448e7d53ca6deeaccab519005110/miniRekog/models/MNISTModels.py#L314). 
Training and test code are available in this [repo](https://github.com/rajy4683/mini-Rekog.git)
## Model Performance 
Below is a table of various combinations that were attempted

| Norm Type | L1          | L2       | Batch Size | Max Validation Accuracy | Training Link                                     |
| --------- | ----------- | -------- | ---------- | ----------------------- | ------------------------------------------------- |
| GN        | 0           | 0        | 128        | 99.31                   | https://wandb.ai/rajy4683/news4eva4/runs/lu61n6n6 |
| GN        | 0.099455391 | 0        | 123        | 99.32                   | https://wandb.ai/rajy4683/news4eva4/runs/273h0g4m |
| GN        | 0.0001      | 0        | 64         | 99.44                   | https://wandb.ai/rajy4683/news4eva4/runs/3hxm2fb8 |
| GN        | 0.0001      | 0.0004   | 64         | 99.44                   | https://wandb.ai/rajy4683/news4eva4/runs/ebxe95ed |
| GN        | 0.0001      | 0.0004   | 32         | 99.41                   | https://wandb.ai/rajy4683/news4eva4/runs/w9mj9man |
| GN        | 0.0001      | 0.0004   | 16         | 99.32                   | https://wandb.ai/rajy4683/news4eva4/runs/26t91la6 |
| LN        | 0           | 0.00002  | 128        | 99.39                   | https://wandb.ai/rajy4683/news4eva4/runs/2bsy9uej |
| LN        | 0           | 0.006604 | 128        | 99.43                   | https://wandb.ai/rajy4683/news4eva4/runs/2ie8bxvy |
| LN        | 0           | 0.006604 | 32         | 99.41                   | https://wandb.ai/rajy4683/news4eva4/runs/1ocyzwe7 |
| LN        | 0           | 0.006604 | 16         | 99.1                    | https://wandb.ai/rajy4683/news4eva4/runs/28whgxhw |

## Loss and Accuracy Graphs
We baselined the graphs for the following models:
1. Group Normalization + L1 regularization
2. Batch Normalization + L1 +  L2 regularizations
3. Layer Normalization + L2 regularization 

Test Loss
![Test Loss](https://github.com/rajy4683/EVA6/blob/master/EVA6S6/imgs/S6EVA6_TestLoss.png)

Training Loss
![Training Loss](https://github.com/rajy4683/EVA6/blob/master/EVA6S6/imgs/S6EVA6_TrainLoss.png)

Test Accuracy
![Test Accuracy](https://github.com/rajy4683/EVA6/blob/master/EVA6S6/imgs/S6EVA6_TestAccuracy.png)

Training Accuracy
![Training Accuracy](https://github.com/rajy4683/EVA6/blob/master/EVA6S6/imgs/S6EVA6_TrainAccuracy.png)

## Observations and Learnings
1. Group Normalization was found to perform better at lower batch sizes also compared to Batch Normalization and Layer Normalization
