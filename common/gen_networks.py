import torch.utils.data
from torch.nn import functional as F
import torch
import torch.nn as nn

#from common.opt import get_norm_layer, group2feature


def group2onehot(groups, age_group):
    code = torch.eye(age_group)[groups.squeeze()]
    if len(code.size()) > 1:
        return code
    return code.unsqueeze(0)


def group2feature(group, age_group, feature_size):
    onehot = group2onehot(group, age_group)
    return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)

                       
def get_norm_layer(norm_layer, module, **kwargs):
    if norm_layer == 'none':
        return module
    elif norm_layer == 'bn':
        return nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'in':
        return nn.Sequential(
           module,
           nn.InstanceNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'sn':
        return nn.utils.spectral_norm(module, **kwargs)
    else:
        return NotImplementedError


class TaskRouter(nn.Module):

    def __init__(self, unit_count, age_group, sigma):
        super(TaskRouter, self).__init__()

        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)

        self.register_buffer('_unit_mapping', torch.zeros((age_group, conv_dim)))
        start = 0
        for i in range(age_group):
            self._unit_mapping[i, start: start + unit_count] = 1
            start = int(start + (1 - sigma) * unit_count)

    def forward(self, inputs, task_ids):
        mask = torch.index_select(self._unit_mapping, 0, task_ids.long()) \
            .unsqueeze(2).unsqueeze(3)
        inputs = inputs * mask
        print(fg256("blue", 'Router: ', inputs.shape))
        return inputs


class ResidualBlock(nn.Module):

    def __init__(self, unit_count, age_group, sigma):
        super(ResidualBlock, self).__init__()
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), nn.BatchNorm2d(conv_dim))
        self.router1 = TaskRouter(unit_count, age_group, sigma)
        self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), nn.BatchNorm2d(conv_dim))
        self.router2 = TaskRouter(unit_count, age_group, sigma)
        self.relu1 = nn.PReLU(conv_dim)
        self.relu2 = nn.PReLU(conv_dim)

    def forward(self, inputs):
        x, task_ids = inputs[0], inputs[1]
        residual = x
        x = self.router1(self.conv1(x), task_ids)
        x = self.relu1(x)
        x = self.router2(self.conv2(x), task_ids)
        return {0: self.relu2(residual + x), 1: task_ids}


class Upsample(nn.Module):

    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        if x.size(2) < up.size(2):
            x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)

        p = self.conv(torch.cat([x, up], dim=1))
        sc = self.shortcut(x)

        p = p + sc
        p2 = self.conv2(p)
        return p + p2


class AgingModule(nn.Module):

    def __init__(self, age_group, repeat_num=4):
        super(AgingModule, self).__init__()
        layers = []
        sigma = 0.1
        unit_count = 4  # 128
        default_dim = 256  # 512
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(
            nn.Conv2d(default_dim, conv_dim, 1, 1, 0),
            nn.BatchNorm2d(conv_dim),
            nn.PReLU(conv_dim),
        )
        self.router = TaskRouter(unit_count, age_group, sigma)
        for _ in range(repeat_num):
            layers.append(ResidualBlock(unit_count, age_group, sigma))
        self.transform = nn.Sequential(*layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_dim, default_dim, 1, 1, 0),
            nn.BatchNorm2d(default_dim),
            nn.PReLU(default_dim),
        )

        self.up_1 = Upsample(default_dim, default_dim+default_dim//2, default_dim//2)
        self.up_2 = Upsample(default_dim//2, default_dim//2+default_dim//4, default_dim//4)
        self.up_3 = Upsample(default_dim//4, default_dim//4+default_dim//8, default_dim//8)
        self.up_4 = Upsample(default_dim//8, default_dim//8+3, default_dim//8)
        self.conv3 = nn.Conv2d(default_dim//8, 3, 1, 1, 0)
        self.__init_weights()

    def forward(self, input_img, x_2, x_3, x_4, x_id, condition):
        x_id = self.conv1(x_id)
        x_id = self.router(x_id, condition)
        print(fg256("white", x_id.shape))

        inputs = {0: x_id, 1: condition}
        x = self.transform(inputs)[0]  # [1, 14, 14, 14]
        print(fg256("yellow", 'Transform: ', x.shape))
        x = self.conv2(x)
        print(fg256("yellow", 'Conv2: ', x.shape))
        x = self.up_1(x, x_4)
        print(fg256("yellow", 'Up1: ', x.shape))
        x = self.up_2(x, x_3)
        print(fg256("orange", 'Up2: ', x.shape))
        x = self.up_3(x, x_2)
        print(fg256("orange", 'Up3: ', x.shape))
        print(fg256("cyan", 'Input: ', input_img.shape))
        x = self.up_4(x, input_img)
        x = self.conv3(x)
        return input_img + x

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)


class PatchDiscriminator(nn.Module):

    def __init__(self, age_group, conv_dim=64, repeat_num=3, norm_layer='bn'):
        super(PatchDiscriminator, self).__init__()

        use_bias = True
        self.age_group = age_group

        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        sequence = []
        nf_mult = 1

        for n in range(1, repeat_num):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n , 8)
            sequence += [
                get_norm_layer(norm_layer, nn.Conv2d(conv_dim * nf_mult_prev + (self.age_group if n == 1 else 0),
                                                     conv_dim * nf_mult, kernel_size=4, stride=2, padding=1,
                                                     bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** repeat_num, 8)

        sequence += [
            get_norm_layer(norm_layer,
                           nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, kernel_size=4,
                                     stride=1, padding=1,
                                     bias=use_bias)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs, condition):
        x = F.leaky_relu(self.conv1(inputs), 0.2, inplace=True)
        condition = group2feature(condition, feature_size=x.size(2), age_group=self.age_group).to(x)
        return self.main(torch.cat([x, condition], dim=1))


if __name__ == '__main__':
    
    from pytorch_model_summary import summary
    from fabulous.color import fg256
    discriminator = PatchDiscriminator(4, norm_layer='sn', repeat_num=4)  # Params: ~7M

    target_img = torch.randn(1, 3, 96, 96)
    target_label = torch.randn(1).long()
    d_logit = discriminator(target_img, target_label)  # [1, 1, 12, 12]
    print(fg256("orange", summary(discriminator, target_img, target_label, show_input=True)))

    generator = AgingModule(age_group=4)
    source_img = torch.randn(1, 3, 96, 96)
    x_2  = torch.randn(1, 32, 48, 48)
    x_3  = torch.randn(1, 64, 24, 24)
    x_4  = torch.randn(1, 128, 12, 12)

    x_exp = torch.randn(1, 256, 6, 6)
    condition = torch.tensor([0])
    g_source = generator(source_img, x_2, x_3, x_4, x_exp, condition)
    print(fg256("cyan", 'g_source: ', g_source.shape))
#    print(fg256("yellow", summary(generator, source_img, x_2, x_3, x_4, x_id, condition, show_input=True)))

