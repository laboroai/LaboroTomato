# Laboro Tomato: Instance segmentation dataset

<!-- TOC -->

- [Laboro Tomato: Instance segmentation dataset](#laboro-tomato-instance-segmentation-dataset)
  - [Overview](#overview)
    - [About Laboro Tomato](#about-laboro-tomato)
    - [Annotation details](#annotation-details)
    - [Dataset details](#dataset-details)
    - [Scope of application](#scope-of-application)
    - [License](#license)
    - [Download dataset](#download-dataset)
  - [Baseline](#baseline)
    - [Pretrained model](#pretrained-model)
      - [Output examples](#output-examples)
  - [Usage Guide](#usage-guide)
    - [MMDetection v3.x](#mmdetection-v3x)
    - [MMDetection v2.x](#mmdetection-v2x)
  - [Subsets](#subsets)
    - [Details](#details)
    - [Pretrained models](#pretrained-models)
    - [Download subsets](#download-subsets)

## Overview

### About Laboro Tomato

Laboro Tomato is an image dataset of growing tomatoes at different stages of their ripening which is designed for object detection and instance segmentation tasks. We also provide two subsets of tomatoes separated by size. Dataset was gathered at a local farm with two separate cameras with its different resolution and image quality.

<img src="https://github.com/laboroai/LaboroTomatoDatasets/blob/master/examples/ann_gif_IMG_1066.gif" width="45%"></img>
<img src="https://github.com/laboroai/LaboroTomatoDatasets/blob/master/examples/ann_gif_IMG_1246.gif" width="45%"></img>
Samples of raw/annotated images: IMG_1066, IMG_1246

### Annotation details

Each tomato is divided into 2 categories according to size (normal size and cherry tomato) and 3 categories depending on the stage of ripening:  

- fully_ripened - completely red color and ready to be harvested. Filled with red color on 90%* or more
- half_ripened - greenish and needs time to ripen. Filled with red color on 30-89%*
- green - completely green/white, sometimes with rare red parts. Filled with red color on 0-30%*

*All percentages are approximate and differ from case to case.

<img src="https://github.com/laboroai/LaboroTomatoDatasets/blob/master/examples/laboro_tomato_exp1.png"></img>

### Dataset details  

Dataset includes 804 images with following details:  

```
name: tomato_mixed
images: 643 train, 161 test
cls_num: 6
cls_names: b_fully_ripened, b_half_ripened, b_green, l_fully_ripened, l_half_ripened, l_green
total_bboxes: train[7781], test[1,996]
bboxes_per_class:
    *Train: b_fully_ripened[348], b_half_ripened[520], b_green[1467], 
            l_fully_ripened[982], l_half_ripened[797], l_green[3667]
    *Test:  b_fully_ripened[72], b_half_ripened[116], b_green[387], 
            l_fully_ripened[269], l_half_ripened[223], l_green[929]
image_resolutions: 3024x4032, 3120x4160
```

<img src="https://github.com/laboroai/LaboroTomatoDatasets/blob/master/examples/laboro_tomato_exp2.png"></img>

### Scope of application

Laboro Tomato dataset can be used to solve cutting edge real-life tasks by fusing various technologies:  

- Harvesting forecast based on tomato maturity
- Automatic harvest of only ripened tomatoes
- Identification and automatic thinning of deteriorated and obsolete tomatoes
- Spraying pesticides only on tomatoes at a specific ripening stage
- Temperature control in greenhouse according to ripening stage
- Quality control on production line of food manufactures, etc.

### License

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png" width="100" /></a></br>

   This work is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.  
   For commercial use, please [contact Laboro.AI Inc.](https://laboro.ai/contact/other/)

### Download dataset

Laboro Tomato download [link](http://assets.laboro.ai/laborotomato/laboro_tomato.zip)  

## Baseline

### Pretrained model

Model have been trained by [MMDetection](https://github.com/open-mmlab/mmdetection) V2.0 on 4 Tesla-V100 and based on Mask R-CNN with R-50-FPN 1x backbone:

|  Dataset      |  bbox AP  |   mask AP    |    Download     |
|---------------|:---------:|:------------:|:----------------:|
|Laboro Tomato   | 64.3 | 65.7 | [model](http://assets.laboro.ai/laborotomato/laboro_tomato_48ep.pth) |

We haven't done hyperparameters tuning for baseline model training and used default values, provided by original MMDetection configs.  
Training parameters:  

``` text
lr = 0.01
step = [32, 44]
total epoch = 48
```

#### Output examples

Image [gallery](https://github.com/laboroai/LaboroTomatoDatasets/blob/master/examples/README.md) with pretrained model output examples and its comparison between raw and annotated images.

## Usage Guide

### MMDetection v3.x
Please refer to the [user guide MMDetection 3.x](usage_guide/usage_guide_mmdetection_v3.md) for information on how to train or test on LaboroTomato dataset by using MMDetection version 3.x.

### MMDetection v2.x
MMDetection version 2.x user can refer to [user guide MMDetection 2.x](usage_guide/usage_guide_mmdetection_v2.md).

## Subsets

### Details

```
name: tomato_big
images: 353 train, 89 test
cls_num: 3
cls_names: b_fully_ripened, b_half_ripened, b_green
total_bboxes: train[2360], test[550]
bboxes_per_class:
    *Train: b_fully_ripened[343], b_half_ripened[506], b_green[1511], 
    *Test:  b_fully_ripened[77], b_half_ripened[130], b_green[343], 
image_resolutions: 3024x4032, 3120x4160
```

```
name: tomato_little
images: 289 train, 73 test
cls_num: 3
cls_names: l_fully_ripened, l_half_ripened, l_green
total_bboxes: train[5397], test[1470]
bboxes_per_class:
    *Train: l_fully_ripened[963], l_half_ripened[805], l_green[3629], 
    *Test:  l_fully_ripened[288], l_half_ripened[215], l_green[967], 
image_resolutions: 3024x4032, 3120x4160
```

### Pretrained models

As well as main dataset, Laboro tomato big and Laboro tomato little have been trained by [MMDetection](https://github.com/open-mmlab/MMDetection) V2.0 on 4 Tesla-V100 and based on Mask R-CNN with R-50-FPN 1x backbone:

|  Dataset      |  bbox AP  |   mask AP    |    Download     |
|---------------|:---------:|:------------:|:----------------:|
|Laboro tomato big     | 67.9 | 68.4 | [model](http://assets.laboro.ai/laborotomato/laboro_tomato_big_48ep.pth) |
|Laboro tomato little  | 62.7 | 63.1 | [model](http://assets.laboro.ai/laborotomato/laboro_tomato_little_48ep.pth) |

Training parameters:  

```
lr = 0.01
step = [32, 44]
total epoch = 48
```

### Download subsets

- [Laboro tomato big](http://assets.laboro.ai/laborotomato/laboro_tomato_big.zip)  
- [Laboro tomato little](http://assets.laboro.ai/laborotomato/laboro_tomato_little.zip)  
