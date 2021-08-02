# Yolov3 Object Detection

In this repo, we will use YoloV3 to perform object detection on two different activities:

1. Task 1: A random image which has classes from COCO dataset
2. Task 2: A new dataset created from scratch that contains custom classes.

## Task 1

The code for this task can be found in this [notebook](https://github.com/rajy4683/EVA6/blob/master/EVA6S11/EVA6S11_Part1.ipynb). Most of the implementation is similar to [this link](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

Sample output of training is below:

![OpenCV Yolo](https://github.com/rajy4683/EVA6/blob/master/EVA6S11/imgs/Annotated_output.JPG)

## Task 2

In this task we train YoloV3 with base weights but with custom datasets that has the following classes:

1. hardhats
2. masks
3. vests
4. boots

The training code is available in [this notebook](https://github.com/rajy4683/EVA6/blob/master/EVA6S11/EVA6_S11YoloV3.ipynb).

### Results

![Part 1](https://github.com/rajy4683/EVA6/blob/master/EVA6S11/imgs/EVA6_Output1.JPG)

![Part 2](https://github.com/rajy4683/EVA6/blob/master/EVA6S11/imgs/EVA6_Output2.JPG)

## References

1. [YoloV3 Training Repo](https://github.com/theschoolofai/YoloV3)
2. Article on [OpenCV Yolo Training](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

