## Introduction

*Our method only adopts the clustering algorithm and ReID baseline model with the moving average model.

UDAStrongBaseline is a transitional code based pyTorch framework for both unsupervised learning (USL) 
and unsupervised domain adaptation (UDA) in the context of object re-ID tasks. It provides stronger 
baselines on these tasks. It needs the enviorment: Python >=3.6 and PyTorch >=1.1. We will transfer all the codes to the [fastreid](https://github.com/JDAI-CV/fast-reid) in the future (ongoing).


### Unsupervised domain adaptation (UDA) on Person re-ID

- `Direct infer` models are trained on the source-domain datasets 
([source_pretrain]()) and directly tested on the target-domain datasets.
- UDA methods (`MMT`, `SpCL`, etc.) starting from ImageNet means that they are trained end-to-end 
in only one stage without source-domain pre-training. `MLT` denotes to the implementation of our NeurIPS-2020. 
Please note that it is a pre-released repository for the anonymous review process, and the official 
repository will be released upon the paper published.

#### DukeMTMC-reID -> Market-1501

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | 
| Direct infer | ResNet50 | DukeMTMC | 32.2 | 64.9 | 78.7 | 83.4 | ~1h | 
| [UDA_TP](https://github.com/open-mmlab/OpenUnReID/) PR'2020| ResNet50 | DukeMTMC | 52.3 | 76.0 | 87.8 | 91.9 | ~2h | 
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020| ResNet50 | imagenet | 80.9 | 92.2 | 97.6 | 98.4 | ~6h |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission| ResNet50 | imagenet | 78.2 | 90.5 | 96.6 | 97.8 | ~3h |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | imagenet | 75.6 | 90.9 | 96.6 | 97.8 | ~3h | 
| [Our stronger_baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | DukeMTMC | 78.0 | 91.0 | 96.4 | 97.7 | ~3h |
| [MLT] NIPS'2020 submission| ResNet50 | DukeMTMC | 81.5| 92.8| 96.8| 97.9 | ~ |

#### Market-1501 -> DukeMTMC-reID

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | 
| Direct infer | ResNet50 | Market | 34.1 | 51.3 | 65.3 | 71.7 | ~1h | 
| [UDA_TP](https://github.com/open-mmlab/OpenUnReID/) PR'2020| ResNet50 | Market1501 | 45.7 | 65.5 | 78.0 | 81.7 | ~2h |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020| ResNet50 | imagenet | 67.7 | 80.3 | 89.9 | 92.9 | ~6h |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission | ResNet50 | imagenet | 70.4 | 83.8 | 91.2 | 93.4 | ~3h |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | imagenet | 60.4 | 75.9 | 86.2 | 89.8 | ~3h |
| [Our stronger_baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | Market1501 | 66.7 | 80.0 | 89.2 | 92.2  | ~3h |
| [MLT] NIPS'2020 submission| ResNet50 | Market1501 | 71.2 |83.9| 91.5| 93.2| ~ |

## Requirements

### Installation

```shell
git https://github.com/zkcys001/UDAStrongBaseline/
cd UDAStrongBaseline

```

### Prepare Datasets

```shell
```
Download the person datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), Then unzip them under the directory like
```
./data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
    └── MSMT17_V1

```

You can create the soft link to the dataset:
```shell
ln -s /path-to-data ./data
```

ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.


## Training

We utilize 4 GPUs for training. **Note that**


### Stage I: Pretrain Model on Source Domain
To train the model(s) in the source domain, run this command:
```shell
sh scripts/pretrain_market1501.sh
sh scripts/pretrain_dukemtmc.sh
```


### Stage II: End-to-end training with clustering

Utilizeing DBSCAN clustering algorithm

```shell
sh scripts/dbscan_baseline_market2duke.sh
sh scripts/dbscan_baseline_duke2market.sh
```





## Acknowledgement

Some parts of `UDAstrongbaseline` are from [MMT](https://github.com/yxgeee/MMT) 
and [fastreid](https://github.com/JDAI-CV/fast-reid). We would like to thank for these projects, 
and we will update our method .
