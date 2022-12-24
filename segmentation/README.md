# Applying FCViT to Semantic Segmentation

Our semantic segmentation implementation is based on [MMSegmentation v0.19.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.19.0) and [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation). Thank the authors for their wonderful works.

## Usage

Install MMSegmentation v0.19.0. 


## Data preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Results and models

| Method | Backbone | Pretrain | Iters | mIoU | Config | Download |
| --- | --- | --- |:---:|:---:| --- | --- |
| Semantic FPN | FCViT-B12   | ImageNet-1K |  40K  |     43.3    | [config](configs/sem_fpn/fpn_fcvit_b12_ade20k_40k.py) | [log & model](https://drive.google.com/drive/folders/17j_ZsGcoavAPC4OdPGXLo8ZEk0R8BZWK?usp=sharing) |
| Semantic FPN | FCViT-B24  | ImageNet-1K |  40K  |     45.5    | [config](configs/sem_fpn/fpn_fcvit_b24_ade20k_40k.py) | [log & model](https://drive.google.com/drive/folders/1mzFxxMxe3XBQEibgrryuBVV7nQXDELKB?usp=sharing) |


## Evaluation
To evaluate FCViT-B12 + Semantic FPN on a single node with 8 GPUs run:
```
dist_test.sh configs/sem_fpn/fpn_fcvit_b12_ade20k_40k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train FCViT-B12 + Semantic FPN on a single node with 8 GPUs run:

```
dist_train.sh configs/sem_fpn/fpn_fcvit_b12_ade20k_40k.py 8
```
