import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, surrogate
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode, MultiStepIFNode



surrogate_function = surrogate.ATan()

__all__ = ['MSResNet', 'ms_resnet18', 'ms_resnet34', 'ms_resnet104']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

    def forward(self, x):
        identity = x

        out = self.conv1(self.sn1(x))

        out = self.conv2(self.sn2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

    def forward(self, x):
        identity = x

        out = self.conv1(self.sn1(x))

        out = self.conv2(self.sn2(out))

        out = self.conv3(self.sn3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class MSResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, ):
        super(MSResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)

        self.sn1 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')
        self.maxpool = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch'),
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sn1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(dim=0)

    def forward(self, x, return_feature=False):
        return self._forward_impl(x, return_feature)

class BasicBlock1(nn.Module):

    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_features / self.in_features == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")

        self.conv1 = nn.Conv2d(in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(_features, _features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # 下采样
        self.downsample = None if self.in_features == self.out_features else nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 输入输出的特征数不同时使用下采样层
        if self.in_features != self.out_features:
            identity = self.downsample(x)

        # 残差求和
        out += identity
        out = self.relu(out)
        return out


class BasicBlockSNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, tau=2.0):
        super(BasicBlockSNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        stride = 1
        _channels = out_channels
        if self.in_channels != self.out_channels:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_channels / self.in_channels == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")
        self.conv1 = layer.Conv2d(in_channels, _channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(_channels)
        # self.lif1 = neuron.LIFNode(tau=tau)

        self.conv2 = layer.Conv2d(_channels, _channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(_channels)
        # self.lif2 = neuron.LIFNode(tau=tau)
        self.relu = nn.ReLU(inplace=True)
        # 下采样层（如果需要）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.lif1(out)
        out = self.relu(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.lif2(out)
        out = self.relu(x)
        return out

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # <---- 输出为{Tensor:(64,512,1,1)}
        x = torch.flatten(x, 1)  # <----------------这里是个坑 很容易漏 从池化层到全连接需要一个压平 输出为{Tensor:(64,512)}
        x = self.fc(x)  # <------------ 输出为{Tensor:(64,10)}
        return x
def ms_resnet(block, layers, **kwargs):
    model = MSResNet(block, layers, **kwargs)
    return model


def ms_resnet18(**kwargs):
    return ms_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ms_resnet34(**kwargs):
    return ms_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ms_resnet104(**kwargs):
    return ms_resnet(BasicBlock, [3, 8, 32, 8], **kwargs)
