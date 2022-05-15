import torch
import torch.nn as nn

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from torchmetrics.functional import dice_score
from torchmetrics.functional import mean_squared_error

from torchvision import transforms

from model.unet import UNetWithResNet
from data.coco_dataset import COCOStuff10kDataset

from util.constants import NUM_OF_CLASSES, ANNOTATIONS_PATH, IMAGES_PATH, TRAIN_IMAGES_LIST_PATH, TEST_IMAGES_LIST_PATH


class LightningUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = UNetWithResNet(NUM_OF_CLASSES)
        self.loss = nn.CrossEntropyLoss()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ])

        train_val_dataset = COCOStuff10kDataset(
            annotations_path=ANNOTATIONS_PATH,
            images_path=IMAGES_PATH,
            images_list_path=TRAIN_IMAGES_LIST_PATH,
            transform=self.transform
        )
        train_size = int(0.8 * len(train_val_dataset))
        validation_size = len(train_val_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_val_dataset, [train_size, validation_size]
        )

        self.test_dataset = COCOStuff10kDataset(
            annotations_path=ANNOTATIONS_PATH,
            images_path=IMAGES_PATH,
            images_list_path=TEST_IMAGES_LIST_PATH,
            transform=self.transform
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def training_step(self, batch, _):
        img, cpt = batch
        outputs = self.model(img)

        loss = self.loss(outputs, cpt)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss
        }

    def validation_step(self, batch, _):
        img, cpt = batch
        outputs = self.model(img)

        loss = self.loss(outputs, cpt)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, _):
        img, cpt = batch
        outputs = self.model(img)
        self.log("dice_score", dice_score(outputs, cpt), prog_bar=True)
        self.log("mse", mean_squared_error(
            outputs.argmax(dim=1), cpt), prog_bar=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=8)
