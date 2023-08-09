
# Usage guide mmdetection 3.x

We will guide you on using mmdetection 3.x to train, test, and evaluate a model on laboro tomato dataset.

## Installation

### Requirements

Ubuntu Linux with Python ≥ 3.8

### Install mmdetection

Please prepare mmdetection environment by official [installation guide](https://mmdetection.readthedocs.io/en/dev-3.x/get_started.html).

## Prepare dataset

### Download dataset

It is recommended to symlink the dataset root to $MMDETECTION/data. If your folder structure is different, you may need to change the corresponding paths in config files.

``` text
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── laboro_tomatodat
│   │   ├── annotations
│   │   ├── train
│   │   ├── test
```

### Download pre_trained model

It is recommended to symlink the model root to $MMDETECTION/pre_trained_model. If your folder structure is different, you may need to change the corresponding paths in config files.

``` text
mmdetection
├── mmdet
├── tools
├── configs
├── data
├── pre_trained_model
│   ├── laboro_tomato_48ep.pth 

```

### Add datasets to mmdetection

Create `laboro_tomato.py` in `mmdet/datasets/` with content of [dataset coco](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/coco.py) configuration file, change class name to `LaboroTomato`, and change `METAINFO` parameter as following example:

``` python

@DATASETS.register_module()
class LaboroTomato(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('b_fully_ripened', 'b_half_ripened', 'b_green', 'l_fully_ripened', 'l_half_ripened', 'l_green'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)]
    }
```

And add dataset names to `mmdet/datasets/__init__.py`:

``` python
from .laboro_tomato import LaboroTomato

__all__ = [    
           ..., 'LaboroTomato'
          ]

```

## Configuration files

Configuration files setup on Tomato Mixed dataset example:  

1. Create `laboro_tomato_coco_instance.py` in `configs/_base_/datasets/` with content of [coco instance](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/_base_/datasets/coco_instance.py) configuration file and change dataset_type, data_root and path of dataloader like the follow example:

``` python
dataset_type = 'LaboroTomato'
data_root = 'data/laboro_tomato/'
...
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/')
        ...
    ))
val_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        ...
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    ...
    ann_file=data_root + 'annotations/test.json',
    ...
    )
test_evaluator = val_evaluator
```

2. Create `laboro_tomato_mask-rcnn_r50_fpn.py` in `configs/_base_/models/` with content of [mask-rcnn_r50_fpn.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/models/faster_rcnn_r50_fpn.py) configuration file and change setting `num_classes`:

``` python
...
model = dict(
    ...
    roi_head=dict(
        ...
        bbox_head=dict(
            ...
            num_classes=6,
            ...
        ),
        ...
        mask_head=dict(
            ...
            num_classes=6,
            ...
        )
        ...
    )
    ...
)
```

3. Create `laboro_tomato_mask-rcnn_r50_fpn_1x_coco.py` in `configs/mask_rcnn/` with content of [mask_rcnn_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) configuration file and change path setting:

``` python
_base_ = [
    '../_base_/models/laboro_tomato_mask-rcnn_r50_fpn.py',
    '../_base_/datasets/laboro_tomato_coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

```

## Evaluation

You can use the following commands to test a dataset:

``` shell
# single-gpu testing
python tools/test.py configs/mask_rcnn/laboro_tomato_mask-rcnn_r50_fpn_1x_coco.py \
                     pre_trained_model/laboro_tomato_48ep.pth --show

# multi-gpu testing with 4 GPUs
./tools/dist_test.sh configs/mask_rcnn/laboro_tomato_mask-rcnn_r50_fpn_1x_coco.py \
                     pre_trained_model/laboro_tomato_48ep.pth 4 --out results.pkl --eval bbox segm                     
```

## Train a model

To train your model finish all steps from _Test a model_ section and change learning rate and total epoch, steps at `configs/_base_/schedules/schedule_1x.py`. The default learning rate in config files is for 8 GPUs and 2 img/gpu (`base_batch_size` = 16 (8x2)). According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs x 2 img/gpu and lr=0.08 for 16 GPUs x 4 img/gpu.

``` python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
```

You can use the following commands to train a model:

``` shell
# single-gpu train
python tools/train.py configs/mask_rcnn/laboro_tomato_mask-rcnn_r50_fpn_1x_coco.py \
                      --work-dir ./laboro_tomato

# multi-gpu train with 4 GPUs
./tools/dist_test.sh configs/mask_rcnn/laboro_tomato_mask-rcnn_r50_fpn_1x_coco.py 4 \
                      --work-dir ./laboro_tomato                    
```
