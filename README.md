# Laboro Tomato: Instance segmantation dataset

<!-- TOC -->

- [Overview](#overview)
    - [About Laboro Tomato](#about-laboro-tomato)
    - [Annotation details](#annotation-details)
    - [Dataset details](#dataset-details)
    - [Scope of application](#scope-of-application)
    - [Licence](#licence)
    - [Download dataset](#download-dataset)
- [Baseline](#baseline)
    - [Pretrained model](#pretrained-model)
        - [Output examples](#output-examples)
    - [Test a dataset](#test-a-dataset)
        - [Prepare dataset](#prepare-dataset)
        - [Add datasets to mmdetection](#add-datasets-to-mmdetection)
        - [Configuration files](#configuration-files)
        - [Evaluation](#evaluation)
    - [Train a model](#train-a-model)
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
* fully_ripened - complitely red color and ready to be harvested. Filled with red color on 90%* or more
* half_ripened - greenish and needs time to ripen. Filled with red color on 30-89%*
* green - complitely green/white, sometimes with rare red parts. Filled with red color on 0-30%*

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
* Harvesting forecast based on tomato maturity
* Automatic harvest of only ripened tomates
* Identification and automatic thinning of deteriorated and obsolete tomatoes 
* Splayig pesticides only on tomatoes at a specific ripening stage 
* Temperature control in greenhouse according to ripening stage
* Quality control on production line of food manufactures, etc.

### Licence 

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png" width="100" /></a></br>

   This work is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.  
   For commercial use, please [contact Laboro.AI Inc.](https://laboro.ai/contact/other/)


### Download dataset

Laboro Tomato download [link](http://assets.laboro.ai/laborotomato/laboro_tomato.zip)  


## Baseline


### Pretrained model 

Model have been trained by [mmdetection](https://github.com/open-mmlab/mmdetection) V2.0 on 4 Tesla-V100 and based on Mask R-CNN with R-50-FPN 1x backbone:
 
|  Dataset      |  bbox AP  |   mask AP    |    Download     |
|---------------|:---------:|:------------:|:----------------:|
|Laboro Tomato   | 64.3 | 65.7 | [model](http://assets.laboro.ai/laborotomato/laboro_tomato_48ep.pth) |

We haven't done hyperparameters tuning for baseline model training and used default values, provided by original mmdetection configs.  
Training parameters:  
```
lr = 0.01
step = [32, 44]
total epoch = 48
```
#### Output examples

Image [gallery](https://github.com/laboroai/LaboroTomatoDatasets/blob/master/examples/README.md) with pretrained model output examples and its comparison between raw and annotated images.


### Test a dataset

To evaluate pretrained models please prepare mmdetection environment by official installation [guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md). 


#### Prepare dataset

It is recommended to symlink the dataset root to $MMDETECTION/data. If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── laboro_tomato
│   │   ├── annotations
│   │   ├── train
│   │   ├── test
```

#### Add datasets to mmdetection

To load data we need to create a new config file `mmdet/datasets/laboro_tomato.py` with corresponding subsets:

```
from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class LaboroTomato(CocoDataset):
    CLASSES = ('b_fully_ripened', 'b_half_ripened', 'b_green', 
               'l_fully_ripened', 'l_half_ripened', 'l_green')
```

And add dataset names to `mmdet/datasets/__init__.py`:

```
from .laboro_tomato import LaboroTomato

__all__ = [    
           ..., 'LaboroTomato'
          ]

```

#### Configuration files

Configuration files setup on Tomato Mixed dataset example:  

1. Create `laboro_tomato_base.py` in `configs/_base_/datasets/` with content of [coco_detection](https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_detection.py) configuration file and change dataset type, root and path parameters:

```
dataset_type = 'LaboroTomato'
data_root = 'data/laboro_tomato/'
...
```

2. Create `laboro_tomato_instance.py` in  `configs/_base_/datasets/` with content of [coco_instance](https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_instance.py) and replace it with your base detection configuration file:

```
_base_ = 'laboro_tomato_base.py'
...
```

3. Replace class numbers at model configuration file `configs/_base_/models/mask_rcnn_r50_fpn.py`:

```
...
num_classes = 6
...
num_classes = 6
...
```

4. Replace dataset configuration file name in `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py` with created at step 3:

```
_base_ = [
    ...
    '../_base_/datasets/laboro_tomato_instance.py',
    ...
]
```

#### Evaluation

You can use the following commands to test a dataset:

```
# single-gpu testing
python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
                     laboro_tomato_96ep.pth --show

# multi-gpu testing with 4 GPUs
./tools/dist_test.sh configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
                     laboro_tomato_96ep.pth 4 --out results.pkl --eval bbox segm                     
```

### Train a model

To train your model finish all steps from *Test a model* section and change learning rate and total epoch, steps at `configs/_base_/schedules/schedule_1x.py`. The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16). According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.

```
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
...
    step=[64, 88])
total_epochs = 96
```

You can use the following commands to train a model:

```
# single-gpu train
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
                      --work-dir ./laboro_tomato

# multi-gpu train with 4 GPUs
./tools/dist_test.sh configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py 4 \
                      --work-dir ./laboro_tomato                    
```

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

As well as main dataset, Laboro tomato big and Laboro tomato little have been trained by [mmdetection](https://github.com/open-mmlab/mmdetection) V2.0 on 4 Tesla-V100 and based on Mask R-CNN with R-50-FPN 1x backbone:

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

### Download subests

* [Laboro tomato big](http://assets.laboro.ai/laborotomato/laboro_tomato_big.zip)  
* [Laboro tomato little](http://assets.laboro.ai/laborotomato/laboro_tomato_little.zip)  
