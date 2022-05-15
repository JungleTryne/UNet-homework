import torch
import torch.nn as nn

from model.resnet import ResNet50
from model.upscale import UpscalePart


class UNetWithResNet(nn.Module):
    def __init__(self, num_classes: int):
        super(UNetWithResNet, self).__init__()

        self.resnet = ResNet50()
        self.upscale = UpscalePart(num_classes)

        self.connecting_conv0 = nn.Conv2d(
            64, 128, kernel_size=1, stride=1, padding=0)
        self.connecting_conv1 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0)
        self.connecting_conv2 = nn.Conv2d(
            512, 512, kernel_size=1, stride=1, padding=0)
        self.connecting_conv3 = nn.Conv2d(
            1024, 1024, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.max_pool(x)

        layers_results = []
        for module in self.resnet.res_layers:
            layers_results.append(x)
            x = module(x)

        x = self.upscale.bn1(self.upscale.conv1(x))
        x = self.upscale.bn2(self.upscale.conv2(
            torch.cat((x, self.connecting_conv3(layers_results[3])), dim=1)))
        x = self.upscale.bn3(self.upscale.conv3(
            torch.cat((x, self.connecting_conv2(layers_results[2])), dim=1)))
        x = self.upscale.bn4(self.upscale.conv4(
            torch.cat((x, self.connecting_conv1(layers_results[1])), dim=1)))
        x = self.upscale.upsample(
            torch.cat((x, self.connecting_conv0(layers_results[0])), dim=1))
        x = self.upscale.conv5(x)

        return x
