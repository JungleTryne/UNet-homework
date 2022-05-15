import torch
import torch.nn as nn


class ResNetBottleneck(nn.Module):

    all_modes = ['identity', 'projection']

    def __init__(self, in_channels: int, out_channels: int, mode='identity'):
        super(ResNetBottleneck, self).__init__()

        self.stride = 1 if mode == 'identity' else 2

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(4 * out_channels)

        self.projection = nn.Identity()
        if mode == 'projection':
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    4 * out_channels,
                    kernel_size=1,
                    stride=self.stride
                ),
                nn.BatchNorm2d(4 * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        old_x = x

        out = x
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.projection(old_x) + self.bn3(self.conv3(out)))
        return out
