from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from ..layers import (
    IBN,
    Non_local,
    get_norm,
)

from .gem_pooling import GeneralizedMeanPoolingP

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }


    def __init__(self, depth, mb_h=2048, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool) # no relu
            
        with_nl=True
        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4
        layers= {34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
        non_layers = {34: [3, 4, 6, 3], 50: [0, 2, 3, 0], 101: [0, 2, 9, 0]}[depth]
        num_splits=1
        if with_nl:
            self._build_nonlocal(layers, non_layers, 'BN', num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # print("w/o GeneralizedMeanPoolingP")
        # self.gap = nn.AdaptiveAvgPool2d(1)
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)
      

        self.memorybank_fc = nn.Linear(2048, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)


        # self.memorybank_fc = nn.Sequential(
        #     nn.Linear(2048, 512, bias=True),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 128, bias=False),
        #     nn.BatchNorm1d(128)
        # )

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes is not None:
                for i,num_cluster in enumerate(self.num_classes):
                    exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                    exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster))
        sour_class=751
        self.classifier_ml =  nn.Sequential(
            nn.Linear(self.num_features, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, sour_class, bias=False),
            nn.BatchNorm1d(sour_class)
        )


        if not pretrained:
            self.reset_params()
            
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
    def forward(self, x, feature_withbn=False, training=False, cluster=False):
        x = self.base(x)
        
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

        if self.cut_at_pooling:return x#FALSE

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))#FALSE
        else:
            bn_x = self.feat_bn(x)#1

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        prob = []
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
        else:
            return x, bn_x

        if feature_withbn:#False
           return bn_x, prob
        mb_x = self.mbn(self.memorybank_fc(bn_x))

        ml_x = self.classifier_ml(bn_x)

        return x, prob, mb_x, ml_x



    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())



def resnet50_sbs(mb_h,**kwargs):
    return ResNet(50, mb_h=mb_h, **kwargs)

