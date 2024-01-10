# ECIENet

## Introduction

This is the PyTorch implementation of our soon paper.

Our training weights will be provided after the paper is accepted!

## Overview

- [Installation](#installation)
- [Getting Started](#getting-started)

## Installation

We test this repository on one Nvidia 2080Ti GPU.

Ubuntu20.04

python=3.6.13

torch=1.8.1


## Getting Started

### Dataset Preparation

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
│── KITTI
│   │── ImageSets
│   │   ├──train.txt & val.txt & trainval.txt & test.txt
│   │── training
│   │   ├──calib & image_2 & label_2 & depth_dense
│   │── testing
│   │   ├──calib & image_2

```
* The dense depth files at: [Google Drive](https://drive.google.com/file/d/1mlHtG8ZXLfjm0lSpUOXHulGF9fsthRtM/view?usp=sharing) 

### Training & Testing

* #### Test and evaluate the pretrained models

```
python tools/test.py 
```
* #### Train a model

```
python tools/train_val.py
```
