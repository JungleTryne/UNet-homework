import torch.nn as nn


class UpscalePart(nn.Module):
    def __init__(self, num_classes: int):
        super(UpscalePart, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.ConvTranspose2d(
            1024 + 1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.ConvTranspose2d(
            512 + 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.ConvTranspose2d(
            256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.conv5 = nn.ConvTranspose2d(
            128 + 128, num_classes, kernel_size=1, stride=1)
