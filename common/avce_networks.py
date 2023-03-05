#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DNN models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import pretrainedmodels.utils as utils

from fabulous.color import fg256

#alexnet = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained=None).cuda()
#resnet  = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None).cuda()
alexnet = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet').cuda()
resnet  = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet').cuda()
print(fg256("yellow", 'Successfully loaded INet weights.'))


class Encoder_Alex(nn.Module):
    def __init__(self):
        super(Encoder_Alex, self).__init__()
        self.features = alexnet._features

    def forward(self, x):
        x = self.features(x)
        return x


class Encoder_R18(nn.Module):

    def __init__(self):
        super(Encoder_R18, self).__init__()

        self.conv1 = resnet.conv1
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Regressor_Alex(nn.Module):

    def __init__(self):
        super(Regressor_Alex, self).__init__()
        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 32)
        self.lin1 = nn.Linear(32, 256)
        self.lin2 = nn.Linear(9216, 32)  #TODO(): add independent linear unit
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.va_regressor = nn.Linear(256, 2)

        self.pw_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.mpool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.apool = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)  # shape: [BS, 9216]
        x_btl_1 = self.relu0(self.lin0(self.drop0(x)))
        x_btl_2 = self.relu1(self.lin1(self.drop1(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)

        return x_va, x_btl_1


class Regressor_R18(nn.Module):

    def __init__(self):
        super(Regressor_R18, self).__init__()

        self.avgpool = resnet.avgpool.cuda()
        self.last_linear = resnet.last_linear.cuda()
        self.lin0 = nn.Linear(1000, 32).cuda()
        self.lin1 = nn.Linear(32, 256).cuda()
        self.va_regressor = nn.Linear(256, 2).cuda()

    def forward(self, x):  # [BS, 512, 7, 7]
        x = torch.flatten(self.avgpool(x), 1)  # [BS, 512]
        x = self.last_linear(x)  # [BS, 1000]
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))  # [BS, 32]
#
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)  # [BS, 2]
#        x_va = self.va_regressor(x_btl_1)  # [BS, 2]
        return x_va, x_btl_1


class Regressor_MMx(nn.Module):

    def __init__(self):
        super(Regressor_MMx, self).__init__()

        self.avgpool = resnet.avgpool.cuda()
        self.last_linear = resnet.last_linear.cuda()
        self.lin0 = nn.Linear(64, 32).cuda()
        self.lin1 = nn.Linear(32, 256).cuda()
        self.va_regressor = nn.Linear(256, 2).cuda()

    def forward(self, x):  # [BS, 512, 7, 7]
#        x = torch.flatten(self.avgpool(x), 1)  # [BS, 512]
#        x = self.last_linear(x)  # [BS, 1000]
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))  # [BS, 32]
#
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)  # [BS, 2]
#        x_va = self.va_regressor(x_btl_1)  # [BS, 2]
        return x_va, x_btl_1



class SPRegressor_light(nn.Module):

    def __init__(self):
        super(SPRegressor_light, self).__init__()
#        self.lin1 = nn.Linear(32, 2)
        self.lin1 = nn.Linear(32, 256)
        self.lin2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return 0.5 * torch.tanh(self.lin2(x))
#        return 0.5 * torch.tanh(self.lin1(x))  # Hyper-parameter; linear projection

class Variational_regressor(nn.Module):

    def __init__(self):
        super(Variational_regressor, self).__init__()
        self.lin1 = nn.Linear(32, 64)
        self.lin2 = nn.Linear(64, 8)  # TODO(2021.6.8): Dimension of `variational space`; 8

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return F.relu(self.lin2(x))


def encoder_Alex():
    encoder = Encoder_Alex()
    return encoder
def encoder_R18():
    encoder = Encoder_R18()
    return encoder

def regressor_Alex():
    regressor = Regressor_Alex()
    return regressor
def regressor_R18():
    regressor = Regressor_R18()
    return regressor
def regressor_MMx():
    regressor = Regressor_MMx()
    return regressor

def spregressor():
    spregressor = SPRegressor_light()
    return spregressor
def vregressor():
    vregressor = Variational_regressor()
    return vregressor


if __name__ == "__main__":

    encoder_R18      = encoder_Alex().cuda()
    regressor_R18    = regressor_Alex().cuda()
    encoder_AL       = encoder_Alex().cuda()
    regressor_AL     = regressor_Alex().cuda()

    sp_regressor = spregressor().cuda()
    v_regressor  = vregressor().cuda()
    print('[INFO] Successfully loaded each model.')

    from pytorch_model_summary import summary
    from pytorch_model_summary import summary
    print(fg256("yellow", summary(Encoder_Alex(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
    print(fg256("cyan", summary(Encoder_R18(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
    print(fg256("green", summary(Regressor_Alex(), torch.ones_like(torch.empty(1, 256, 6, 6)), show_input=True)))
    print(fg256("orange", summary(Regressor_R18(), torch.ones_like(torch.empty(1, 512, 8, 8)).cuda(), show_input=True)))
    print(fg256("yellow", summary(SPRegressor_light(), torch.ones_like(torch.empty(1, 32)), show_input=True)))
    print(fg256("green", summary(Variational_regressor(), torch.ones_like(torch.empty(1, 32)), show_input=True)))
