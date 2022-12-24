# Applying FCViT to Object Detection

Our detection implementation is based on [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) and [PVT detection](https://github.com/whai362/PVT/tree/v2/detection). Thank the authors for their wonderful works.



## Usage

Install [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) from souce cocde,

or

```
pip install mmdet==2.19.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0).


## Results and models on COCO


| Backbone       | Parmas | AP-box   | AP-box@50 | AP-box@75 | AP-mask  | AP-mask@50 | AP-mask@75 | Download |
|:--------------:|:------:|:--------:|:---------:|:---------:|:--------:|:----------:|:----------:|:----------:|
| ResNet18       | 31.2M  | 34.0     | 54.0      | 36.7      | 31.2     | 51.0       | 32.7       | |
| PoolFormer-S12 | 31.6M  | 37.3     | 59.0      | 40.1      | 34.6     | 55.8       | 36.9       | |
| PVT-Tiny       | 32.9M  | 36.7     | 59.2      | 39.3      | 35.1     | 56.7       | 37.3       | |
| **FCViT-B12**  | 33.7M  | **42.3** | **64.2**  | **46.2**  | **38.6** | **61.1**   | **41.3**   | [[log & model]](https://drive.google.com/drive/folders/1EAz0qkbGGNqy7SgYr7E3pPCMo8YYnTVK?usp=sharing)|
|----|----|----|----|----|----|----|----|---- |
| ResNet50       | 44.2M  | 38.0     | 58.6      | 41.4      | 34.4     | 55.1       | 36.7       | |
| Poolformer-S24 | 41.0M  | 40.1     | 62.2      | 43.4      | 37.0     | 59.1       | 39.6       | |
| PVT-Small      | 44.1M  | 40.4     | 62.9      | 43.8      | 37.8     | 60.1       | 40.3       | |
| **FCViT-B24**  | 43.1M  | **44.1** | **65.4**  | **48.4**  | **39.9** | **62.4**   | **42.7**   | [[log & model]](https://drive.google.com/drive/folders/1XdZyVcC9oZqNFnUhQWRXNLwoupahMLAp?usp=sharing)|

## Evaluation

To evaluate FCViT-B12 + Mask R-CNN on COCO val2017, run:
```
dist_test.sh configs/mask_rcnn_fcvit_b12_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox segm
```


## Training

To train FCViT-B12 + Mask R-CNN on COCO train2017:
```
dist_train.sh configs/mask_rcnn_fcvit_b12_fpn_1x_coco.py 8
```
