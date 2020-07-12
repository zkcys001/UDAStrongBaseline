# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
from torch.utils import model_zoo
from torch.nn import functional as F
from UDAsbs.layers import (
    IBN,
    Non_local,
    get_norm,
)
from .gem_pooling import GeneralizedMeanPoolingP
from torch.nn import init

model_urls = {
    18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.relu = nn.ReLU(inplace=True)
        # if with_se:
        #     self.se = SELayer(planes, reduction)
        # else:
        self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * 4, num_splits)
        self.relu = nn.ReLU(inplace=True)
        # if with_se:
        #     self.se = SELayer(planes * 4, reduction)
        # else:
        self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block, layers, non_layers,
                 mb_h=2048, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64, num_splits)

        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, num_splits, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, num_splits, with_se=with_se)

        self.random_init()

        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []


        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)

        # self.memorybank_fc=nn.Sequential(
        #     nn.Linear(2048,512,bias=True),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 128,bias=False),
        #     nn.BatchNorm1d(128)
        # )
        #
        self.memorybank_fc = nn.Linear(2048, 2048)
        self.mbn = nn.BatchNorm1d(2048)#get_norm(bn_norm, 2048)#
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)


        self.num_classes = num_classes

        self.dropout = dropout
        out_planes = 2048#resnet.fc.in_features
        self.num_features = out_planes
        self.feat_bn = nn.BatchNorm1d(self.num_features)#get_norm(bn_norm, self.num_features)#

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        for i, num_cluster in enumerate(num_classes):
            exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i, num_cluster,
                                                                                              num_cluster))
            exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i, num_cluster))

        # init.constant_(self.feat_bn.weight, 1)
        # init.constant_(self.feat_bn.bias, 0)
        self.register_buffer("pixel_mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        print("Model Norm")
    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", num_splits=1, with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion, num_splits),
            )

        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, num_splits):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm, num_splits) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm, num_splits) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm, num_splits) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm, num_splits) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x, training =False):
        x = x.sub_(self.pixel_mean).div_(self.pixel_std)
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        bn_x = self.feat_bn(x)

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x


        if self.dropout > 0:bn_x = self.drop(bn_x)

        prob = []

        for i, num_cluster in enumerate(self.num_classes):
            exec("prob.append(self.classifier{}_{}(bn_x))".format(i, num_cluster))


        mb_x = self.mbn(self.memorybank_fc(bn_x))

        return x, prob, mb_x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet50ibn_sbs(mb_h,**kwargs):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = True#cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path ='./logs/pretrained/resnet50_ibn_a.pth.tar'#cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = 1#cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = "syncBN"#cfg.MODEL.BACKBONE.NORM
    num_splits = 1#cfg.MODEL.BACKBONE.NORM_SPLIT
    with_ibn = True#cfg.MODEL.BACKBONE.WITH_IBN
    with_se = False#cfg.MODEL.BACKBONE.WITH_SE
    with_nl = True#cfg.MODEL.BACKBONE.WITH_NL
    depth = 50#cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {34: [3, 4, 6, 3], 50: [0, 2, 3, 0], 101: [0, 2, 9, 0]}[depth]
    block = {34: BasicBlock, 50: Bottleneck, 101: Bottleneck}[depth]
    model = ResNet(last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage, mb_h,**kwargs)

    if pretrain:
        if not with_ibn:
            try:
                state_dict = torch.load(pretrain_path)['model']
                # Remove module.encoder in name
                new_state_dict = {}
                for k in state_dict:
                    new_k = '.'.join(k.split('.')[2:])
                    if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                        new_state_dict[new_k] = state_dict[k]
                state_dict = new_state_dict
                print(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError or KeyError:
                # original resnet
                state_dict = model_zoo.load_url(model_urls[depth])
                print("Loading pretrained model from torchvision")
        else:
            state_dict = torch.load(pretrain_path)['state_dict']  # ibn-net
            # Remove module in name
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])
                if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            print(f"Loading pretrained model from {pretrain_path}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(
                incompatible.missing_keys
            )
        if incompatible.unexpected_keys:
            print(
                incompatible.unexpected_keys
            )
    return model
def resnet50_sbs(mb_h,**kwargs):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = True#cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path =''#''./logs/pretrained/resnet50-19c8e357.pth'#cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = 1#cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = "BN"#cfg.MODEL.BACKBONE.NORM
    num_splits = 1#cfg.MODEL.BACKBONE.NORM_SPLIT
    with_ibn = False#cfg.MODEL.BACKBONE.WITH_IBN
    with_se = False#cfg.MODEL.BACKBONE.WITH_SE
    with_nl = True#cfg.MODEL.BACKBONE.WITH_NL
    depth = 50#cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {34: [3, 4, 6, 3], 50: [0, 2, 3, 0], 101: [0, 2, 9, 0]}[depth]
    block = {34: BasicBlock, 50: Bottleneck, 101: Bottleneck}[depth]
    model = ResNet(last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage, mb_h,**kwargs)

    if pretrain:
        if not with_ibn:
            try:
                state_dict = torch.load(pretrain_path)['model']
                # Remove module.encoder in name
                new_state_dict = {}
                for k in state_dict:
                    new_k = '.'.join(k.split('.')[2:])
                    if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                        new_state_dict[new_k] = state_dict[k]
                state_dict = new_state_dict
                print(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError or KeyError:
                # original resnet
                state_dict = model_zoo.load_url(model_urls[depth])
                print("Loading pretrained model from torchvision")
        else:
            state_dict = torch.load(pretrain_path)['state_dict']  # ibn-net
            # Remove module in name
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])
                if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            print(f"Loading pretrained model from {pretrain_path}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(
                incompatible.missing_keys
            )
        if incompatible.unexpected_keys:
            print(
                incompatible.unexpected_keys
            )
    return model

def resnet50_base(mb_h,**kwargs):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = True#cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path =''#''./logs/pretrained/resnet50-19c8e357.pth'#cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = 1#cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = "BN"#cfg.MODEL.BACKBONE.NORM
    num_splits = 1#cfg.MODEL.BACKBONE.NORM_SPLIT
    with_ibn = False#cfg.MODEL.BACKBONE.WITH_IBN
    with_se = False#cfg.MODEL.BACKBONE.WITH_SE
    with_nl = False#cfg.MODEL.BACKBONE.WITH_NL
    depth = 50#cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {34: [3, 4, 6, 3], 50: [0, 2, 3, 0], 101: [0, 2, 9, 0]}[depth]
    block = {34: BasicBlock, 50: Bottleneck, 101: Bottleneck}[depth]
    model = ResNet(last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage, mb_h,**kwargs)

    if pretrain:
        if not with_ibn:
            try:
                state_dict = torch.load(pretrain_path)['model']
                # Remove module.encoder in name
                new_state_dict = {}
                for k in state_dict:
                    new_k = '.'.join(k.split('.')[2:])
                    if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                        new_state_dict[new_k] = state_dict[k]
                state_dict = new_state_dict
                print(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError or KeyError:
                # original resnet
                state_dict = model_zoo.load_url(model_urls[depth])
                print("Loading pretrained model from torchvision")
        else:
            state_dict = torch.load(pretrain_path)['state_dict']  # ibn-net
            # Remove module in name
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])
                if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            print(f"Loading pretrained model from {pretrain_path}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(
                incompatible.missing_keys
            )
        if incompatible.unexpected_keys:
            print(
                incompatible.unexpected_keys
            )
    return model





