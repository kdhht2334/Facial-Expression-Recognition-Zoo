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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#alexnet   = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained=None).cuda()
#resnet    = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None).cuda()
alexnet   = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet').cuda()
resnet    = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet').cuda()
seresnet  = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet').cuda()
print(fg256("magenta", 'Successfully loaded INet weights.'))


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


class Encoder_SR50(nn.Module):
    def __init__(self):
        super(Encoder_SR50, self).__init__()
        self.features = nasnet

    def forward(self, x):
        x = self.features(x)  # [2, 1000]
        return x


class Regressor_Alex(nn.Module):

    def __init__(self, latent_dim):
        super(Regressor_Alex, self).__init__()
        self.avgpool = alexnet.avgpool
        # AlexNet-original
#        self.lin0 = nn.Linear(9216, 1024)
#        self.lin1 = nn.Linear(1024, 1024)
#        self.lin2 = nn.Linear(1024, 1024)
        # AlexNet-reduced
        self.lin0 = nn.Linear(9216, 64)  # 32
        self.lin1 = nn.Linear(64, latent_dim)  # 256
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.va_regressor = nn.Linear(latent_dim, 2)  # 256

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x_btl_1 = self.relu0(self.lin0(self.drop0(x)))
        x_btl_2 = self.relu1(self.lin1(self.drop1(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_2  # x_btl_1


class Regressor_Alex_Corr(nn.Module):

    def __init__(self):
        super(Regressor_Alex_Corr, self).__init__()
        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 32)
        self.lin_exp = nn.Linear(32, 512)
        self.lin_id  = nn.Linear(32, 512)
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.va_regressor = nn.Linear(512, 2)  # 256

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x_btl_1 = self.relu0(self.lin0(self.drop0(x)))
        x_exp = self.lin_exp(x_btl_1)
        x_id  = self.lin_id(x_btl_1)

        x_btl_2 = self.relu1(self.lin1(self.drop1(x_exp)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_exp, x_id


class Regressor_AL_Category(nn.Module):

    def __init__(self):
        super(Regressor_AL_Category, self).__init__()
        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 32)
        self.lin1 = nn.Linear(32, 256)
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.va_regressor = nn.Linear(256, 7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x_btl_1 = self.relu0(self.lin0(self.drop0(x)))
        x_btl_2 = self.relu1(self.lin1(self.drop1(x_btl_1)))
        x_category = self.sigmoid(self.va_regressor(x_btl_2))
        return x_category, x_btl_1


class Regressor_R18(nn.Module):

    def __init__(self, latent_dim):
        super(Regressor_R18, self).__init__()

        self.avgpool = resnet.avgpool.cuda()
        self.last_linear = resnet.last_linear.cuda()
        self.lin0 = nn.Linear(1000, 64).cuda()  # 32
        self.lin1 = nn.Linear(64, latent_dim).cuda()
        self.va_regressor = nn.Linear(latent_dim, 2).cuda()

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.last_linear(x)
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_2


class Regressor_SR50(nn.Module):

    def __init__(self):
        super(Regressor_SR50, self).__init__()

        self.lin0 = nn.Linear(1000, 32).cuda()
        self.lin1 = nn.Linear(32, 512).cuda()
        self.va_regressor = nn.Linear(512, 2).cuda()

    def forward(self, x):
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_2


class Regressor_R18_Category(nn.Module):

    def __init__(self):
        super(Regressor_R18_Category, self).__init__()

        self.avgpool = resnet.avgpool.cuda()
        self.last_linear = resnet.last_linear.cuda()
        self.lin0 = nn.Linear(1000, 32).cuda()
        self.lin1 = nn.Linear(32, 256).cuda()
        self.va_regressor = nn.Linear(256, 7).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.last_linear(x)
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))

        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        #x_va = self.sigmoid(self.va_regressor(x_btl_2))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_1


class Regressor_R50(nn.Module):

    def __init__(self):
        super(Regressor_R50, self).__init__()

        self.lin0 = nn.Linear(1000, 32).cuda()
        self.lin1 = nn.Linear(32, 256).cuda()
        self.va_regressor = nn.Linear(256, 2).cuda()

    def forward(self, x):
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_1


class Regressor_R101(nn.Module):

    def __init__(self):
        super(Regressor_R101, self).__init__()

        self.lin0 = nn.Linear(1000, 32).cuda()
        self.lin1 = nn.Linear(32, 256).cuda()
        self.va_regressor = nn.Linear(256, 2).cuda()

    def forward(self, x):
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_1


class Regressor_MMx(nn.Module):

    def __init__(self, latent_dim):
        super(Regressor_MMx, self).__init__()

#        self.avgpool = resnet.avgpool.cuda()
#        self.last_linear = resnet.last_linear.cuda()
        self.lin0 = nn.Linear(latent_dim, 256).cuda()
        self.lin1 = nn.Linear(256, latent_dim).cuda()
        self.va_regressor = nn.Linear(latent_dim, 2).cuda()

    def forward(self, x):
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))
        x_btl_2 = F.relu(self.lin1(F.dropout2d(x_btl_1)))
        x_va = self.va_regressor(x_btl_2)
        return x_va, x_btl_2



class SPRegressor_light(nn.Module):

    def __init__(self, discrete_opt):
        super(SPRegressor_light, self).__init__()
        self.discrete_opt = discrete_opt
        self.sigmoid = nn.Sigmoid()
        self.lin1 = nn.Linear(32, 256)
        if self.discrete_opt:
            self.lin2 = nn.Linear(256, 7)
        else:
            self.lin2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        if self.discrete_opt:
            return self.lin2(x)
        else:
            return 0.5 * torch.tanh(self.lin2(x))


class Variational_regressor(nn.Module):

    def __init__(self):
        super(Variational_regressor, self).__init__()
        self.lin1 = nn.Linear(32, 64)
        self.lin2 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return F.relu(self.lin2(x))


class Neural_decomp_mlp(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Neural_decomp_mlp, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_exp = nn.Linear(self.input_dim, self.output_dim)
        self.linear_x = nn.Linear(self.input_dim, self.output_dim)
        self.linear_xxx = nn.Linear(self.input_dim*2, self.output_dim)
        self.bn_exp = nn.BatchNorm1d(self.input_dim, affine=True)
        self.bn_xxx = nn.BatchNorm1d(self.input_dim, affine=True)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self, all_z, z_xxx, option):
        if option == 'simple':
            z_exp = all_z - z_xxx
#            z_exp = self.leakyrelu(self.bn_exp(self.linear_exp(all_z)))
#            z_xxx = self.leakyrelu(self.bn_xxx(self.linear_xxx(torch.cat([all_z, z_xxx], dim=-1))))
            return z_exp, z_xxx
        elif option == 'residual_post':
            all_z = self.leakyrelu(self.bn_exp(self.linear_exp(all_z)))
            z_xxx = self.leakyrelu(self.bn_xxx(self.linear_xxx(torch.cat([all_z, z_xxx], dim=-1))))
            z_exp = all_z - z_xxx
            return z_exp, z_xxx
        elif option == 'residual_prev':
            z_exp = all_z - z_xxx
            z_exp = self.leakyrelu(self.bn_exp(self.linear_exp(z_exp)))
            z_xxx = self.leakyrelu(self.bn_xxx(self.linear_x(z_xxx)))
            return z_exp, z_xxx


class Linear_Prob(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Linear_Prob, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim, self.output_dim*5)
        self.linear2 = nn.Linear(self.output_dim*5, self.output_dim)
        self.bn1 = nn.BatchNorm1d(self.output_dim*5, affine=True)

        self.layer_blocks = nn.Sequential(
            self.linear1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.linear2,
        )

    def forward(self, inp, option):
        if option == 'double':
            all_z, i_clip = inp[0], inp[1]
            concat_z = torch.cat([all_z, i_clip.unsqueeze(1)], dim=-1)
            return self.layer_blocks(concat_z)
        elif option == 'single':
            return self.layer_blocks(inp)


class Linear_Prob_VA(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Linear_Prob_VA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        self.bn1 = nn.BatchNorm1d(self.output_dim, affine=True)

        self.layer_blocks = nn.Sequential(
            self.linear1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.linear2,
        )

    def forward(self, z_exp):
        return self.layer_blocks(z_exp)


class Face_generator_upsample(nn.Module):  # no. of params.: ~4.2M

    def __init__(self):
        super(Face_generator_upsample, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=3),

#            nn.Conv2d(512, 512, 3, stride=1, padding=1),
#            nn.BatchNorm2d(512, 0.8, affine=True),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=3),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 64, 3, stride=1, padding=3),
            nn.BatchNorm2d(64, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, feature_maps):  # feature map shape: [1, 256, 6, 6]
        img = self.conv_blocks(feature_maps)  # img shape: [1, 3, 224, 224]
        return img



class Identity_generator(nn.Module):  # no. of params.: ~554.435

    def __init__(self):
        super(Identity_generator, self).__init__()

        self.init_size = 14  #opt.img_size // 4 # initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(32, 128 * 14 ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):  # noise shape: [1, 10]
        out = self.l1(noise)  # out shape: [1, 2048]
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)  # img shape: [1, 3, 224, 224]
        return img


def encoder_Alex():
    encoder = Encoder_Alex()
    return encoder
def encoder_R18():
    encoder = Encoder_R18()
    return encoder

def regressor_Alex():
    regressor = Regressor_Alex()
    return regressor
def regressor_Alex_Corr():
    regressor = Regressor_Alex_Corr()
    return regressor
def regressor_R18():
    regressor = Regressor_R18()
    return regressor
def regressor_R50():
    regressor = Regressor_R50()
    return regressor
def regressor_R101():
    regressor = Regressor_R101()
    return regressor
def regressor_MMx(latent_dim):
    regressor = Regressor_MMx(latent_dim)
    return regressor
def regressor_AL_Category():
    regressor = Regressor_AL_Category()
    return regressor
def regressor_R18_Category():
    regressor = Regressor_R18_Category()
    return regressor

def spregressor(discrete_opt):
    spregressor = SPRegressor_light(discrete_opt)
    return spregressor
def vregressor():
    vregressor = Variational_regressor()
    return vregressor
def load_Linear_Prob(input_dim, output_dim):
    return Linear_Prob(input_dim, output_dim)
def load_Linear_Prob_VA(input_dim, output_dim):
    return Linear_Prob_VA(input_dim, output_dim)
def load_face_generator():
    return Face_generator_upsample()
    #return Identity_generator()


if __name__ == "__main__":

    from pytorch_model_summary import summary
    print(fg256("yellow", summary(Encoder_Alex(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
    print(fg256("cyan", summary(Encoder_R18(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
    print(fg256("green", summary(Regressor_Alex(), torch.ones_like(torch.empty(1, 256, 6, 6)), show_input=True)))
    print(fg256("orange", summary(Regressor_R18(), torch.ones_like(torch.empty(1, 512, 8, 8)).cuda(), show_input=True)))
    print(fg256("yellow", summary(SPRegressor_light(), torch.ones_like(torch.empty(1, 32)), show_input=True)))
    print(fg256("green", summary(Variational_regressor(), torch.ones_like(torch.empty(1, 32)), show_input=True)))
