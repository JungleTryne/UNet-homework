import os
from pathlib import Path

import cv2 as cv
import scipy.io
from torch.utils.data import Dataset


class COCOStuff10kDataset(Dataset):
    """
    COCOStuff 10k Dataset
    https://github.com/nightrome/cocostuff10k
    """

    def __init__(self, annotations_path: str, images_path: str, images_list_path: str, transform=None):
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.images_list_path = images_list_path

        with open(images_list_path, "r") as f:
            self.images_list = f.read().split()

        self.annotations = [os.path.join(
            annotations_path, image + ".mat") for image in self.images_list]
        self.images = [os.path.join(annotations_path, image + ".jpg")
                       for image in self.images_list]
        assert len(self.annotations) == len(self.images), \
            "Number of annotations and images must be equal: {} != {}".format(
                len(self.annotations), len(self.images))

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        image_name = Path(self.annotations[idx]).stem
        mat = scipy.io.loadmat(self.annotations[idx])["S"]
        img = cv.cvtColor(cv.imread(os.path.join(
            self.images_path, image_name + ".jpg")), cv.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img) * 255
            mat = self.transform(mat) * 255
            mat = mat.long().squeeze()

        return (img, mat)
