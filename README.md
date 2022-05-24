# Object Detection with ConvNext backbone

This repo contains the supported code and configuration files to reproduce object detection results of [ConvNext](https://arxiv.org/pdf/2201.03545.pdf). It is based on [facebookresearch/ConvNeXt/object_detection](https://github.com/facebookresearch/ConvNeXt/tree/main/object_detection).

## Updates
***24/05/2022*** Add train results

***20/05/2022*** Initial commits

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Lr Schd | box mAP | mask mAP | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|
| ConvNeXt-T | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth) | Mask R-CNN | 3x | 46.2 | 41.6 | 48M | 262G | [model](https://dl.fbaipublicfiles.com/convnext/coco/mask_rcnn_convnext_tiny_1k_3x.pth) |



## Usage

### Installation
```
# virtual env

virtualenv convnext

source ./convext/bin/activate


git clone --recursive https://github.com/johnson-magic/ConvNeXt_detection.git

cd ConvNeXt_detection

pip3 install torch torchvision torchaudio

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html

pip install -r requirements.txt

python setup.py develop

git clone https://github.com/NVIDIA/apex

cd apex

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Other Links

> **ConvNeXt**: See [ConvNext](https://github.com/facebookresearch/ConvNeXt).

> **Swin Transformer**: See [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).


