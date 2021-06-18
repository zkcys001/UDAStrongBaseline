# UDAStrongBaseline
Open-source stronger baseline for unsupervised or domain adaptive object re-ID.
We will udpate the strong baseline and group-aware label transfer method in domain adaptive person re-identifacation.

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
| [Our stronger_baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | DukeMTMC | 77.4 | 91.0 | 96.4 | 97.7 | ~3h |
| [Our stronger_baseline + GLT (Kmeans)](https://arxiv.org/pdf/2103.12366.pdf) | ResNet50 | DukeMTMC | 79.5 | 92.7 | 96.9 | 98.0 | ~35h |
| [Our stronger_baseline + uncertainty (DBSCAN)](https://arxiv.org/pdf/2012.08733.pdf) | ResNet50 | DukeMTMC | 82.0 | 93.0 | 97.3 | 98.2 | ~5h |

#### Market-1501 -> DukeMTMC-reID

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | 
| Direct infer | ResNet50 | Market1501 | 34.1 | 51.3 | 65.3 | 71.7 | ~1h | 
| [UDA_TP](https://github.com/open-mmlab/OpenUnReID/) PR'2020| ResNet50 | Market1501 | 45.7 | 65.5 | 78.0 | 81.7 | ~2h |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020| ResNet50 | imagenet | 67.7 | 80.3 | 89.9 | 92.9 | ~6h |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission | ResNet50 | imagenet | 70.4 | 83.8 | 91.2 | 93.4 | ~3h |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | imagenet | 60.4 | 75.9 | 86.2 | 89.8 | ~3h |
| [Our stronger_baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | Market1501 | 66.7 | 80.0 | 89.2 | 92.2  |  ~3h |
| [Our stronger_baseline + uncertainty (DBSCAN)](https://arxiv.org/pdf/2012.08733.pdf) | ResNet50 | Market1501 | 71.8 | 84.0 | 91.7 | 93.8 | ~5h |

## Requirements

### Installation

```shell
git https://github.com/zkcys001/UDAStrongBaseline/
cd UDAStrongBaseline
pip install -r requirements
pip install faiss-gpu==1.6.3
```

### Prepare Datasets

Download the person datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), Then unzip them under the directory like
```
./data
├── dukemtmc
│  └── DukeMTMC-reID
├── market1501
│  └── Market-1501-v15.09.15
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


### 1. Stronger Baseline:

#### Stage I: Pretrain Model on Source Domain
Training the baseline in the source domain, run this command:
```shell
sh scripts/pretrain_market1501.sh
```

#### Stage II: End-to-end training with clustering

Utilizing the baseline based on DBSCAN clustering algorithm:

```shell
sh scripts/dbscan_baseline_market2duke.sh

```

### 2. Uncertainty(AAAI 2021):

#### Stage I: Pretrain Model on Source Domain

Training the uncertainty model in the source domain, run this command:
```shell
sh scripts/pretrain_uncertainty_dukemtmc.sh
```
#### Stage II: End-to-end training with clustering

Utilizing the uncertainty model based on DBSCAN clustering algorithm:
```shell
sh scripts/dbscan_uncertainty_duke2market.sh
```


### 3. GLT (group-aware label transfer, CVPR 2021):

#### Stage I: Pretrain Model on Source Domain
Training the GLT model in the source domain, run this command:
```shell
sh scripts/pretrain_dukemtmc.sh
```

#### Stage II: End-to-end training with clustering
Utilizing the GLT model based on K-means clustering algorithm:
```shell
sh scripts/GLT_kmeans_duke2market.sh
```




## Acknowledgement

Some parts of `UDAstrongbaseline` are from [MMT](https://github.com/yxgeee/MMT) 
and [fastreid](https://github.com/JDAI-CV/fast-reid). We would like to thank for these projects, 
and we will update our method .

## Citation
If you find this code useful for your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{zheng2020exploiting,
     title     =  {Exploiting Sample Uncertainty for Domain Adaptive Person Re-Identification},
     author    =  {Zheng, Kecheng and Lan, Cuiling and Zeng, Wenjun and Zhang, Zhizheng and Zha, Zheng-Jun},
     booktitle =  {Proceedings of the AAAI Conference on Artificial Intelligence},
     year      =  {2021}
}

@InProceedings{Zheng_2021_CVPR,
    author    =   {Zheng, Kecheng and Liu, Wu and He, Lingxiao and Mei, Tao and Luo, Jiebo and Zha, Zheng-Jun},
    title     =   {Group-aware Label Transfer for Domain Adaptive Person Re-identification},
    booktitle =   {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     =   {June},
    year      =   {2021},
    pages     =   {5310-5319}
}
```
