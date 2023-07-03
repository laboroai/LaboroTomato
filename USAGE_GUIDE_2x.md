
# Usage guide mmdetection 2.x

We will guide you on using mmdetection 2.x to train, test, and evaluate a model on laboro tomato dataset.

## Installation

### Requirements

To evaluate pretrained models please prepare mmdetection 2.x environment by official installation [guide](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/get_started.md). 

## Prepare dataset

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

### Add datasets to mmdetection

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

### Configuration files

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

## Evaluation

You can use the following commands to test a dataset:

```
# single-gpu testing
python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
                     laboro_tomato_96ep.pth --show

# multi-gpu testing with 4 GPUs
./tools/dist_test.sh configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
                     laboro_tomato_96ep.pth 4 --out results.pkl --eval bbox segm                     
```

## Train a model

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
