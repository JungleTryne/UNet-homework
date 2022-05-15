import torch
import torch.nn as nn

from model.bottleneck import ResNetBottleneck

from typing import List


class ResNet(nn.Module):
    def __init__(self, configuration: List[int], in_channels: int):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        layers = []
        current_channels = 64
        channels = [64, 128, 256, 512]
        assert len(channels) == len(configuration), \
            "configuration must be the size of {}".format(len(channels))

        for layer_length, channel_size in zip(configuration, channels):
            current_layer = []

            current_layer.append(
                ResNetBottleneck(
                    current_channels,
                    channel_size,
                    mode='projection',
                )
            )

            current_channels = 4 * channel_size
            for _ in range(1, layer_length):
                current_layer.append(
                    ResNetBottleneck(
                        current_channels,
                        channel_size,
                        mode='identity'
                    )
                )

            layers.append(nn.Sequential(*current_layer))

        self.res_layers = nn.Sequential(*layers)
        self.relu = nn.ReLU()


def ResNet50(in_channels: int=3):
    return ResNet([3, 4, 6, 3], in_channels)
