import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from torchvision import transforms
import torchvision.transforms.functional as TF
import itertools
from torch.utils.data.sampler import Sampler


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            labeled_bs=None,
            max_samples_rate=1,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.label_list = []
        self.split = split
        self.transform = transform
        self.LabeledRandomTransform = RandomGenerator()
        self.max_samples_rate = max_samples_rate

        if self.split == "train":
            f1 = h5py.File(base_dir, "r")
            self.sample_list = f1["train_image"]
            self.label_list = f1["train_label"]

        elif self.split == "val":
            f = h5py.File(base_dir, "r")
            self.sample_list = f["val_image"]
            self.label_list = f["val_label"]

    def __len__(self):
        return int(len(self.sample_list) * self.max_samples_rate)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.sample_list[idx]) / 255
        image = image.permute(2, 0, 1)
        image = self.transform(image)

        label = torch.from_numpy(self.label_list[idx])
        if torch.max(label) != 0:
            label = label / torch.max(label)
            label = torch.round(label)  # otherwise, 0 divided by 255 will have error, not completely 0

        # if self.split == "train":  # labeled image data augmentation
        #     image, label = self.LabeledRandomTransform(image, label)

        sample = {"image": image.float(), "label": label.float()}
        # now image[3,256,256], range 0-1, label[256,256], value is 0 or 1

        return sample


# labeled random transforms
class RandomHorizontalFlipWithLabel:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            img = TF.hflip(img)
            label = TF.hflip(label)
        return img, label


class RandomVerticalFlipWithLabel:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            img = TF.vflip(img)
            label = TF.vflip(label)
        return img, label


class RandomRotate:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        # angle = random.random() * 360.0
        angle = random.choice([0, 90, 180, 270])
        img = TF.rotate(img, angle=angle)
        label = TF.rotate(label, angle=angle)
        return img, label


class RandomAffine:
    def __init__(self, p=1,  translate=64, scale=(0.8, 1.2), shear=5):
        self.p = p
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img, label):
        # Random translation
        rand_translate = [int((random.random() - 0.5) * 2 * self.translate), int((random.random() - 0.5) * 2 * self.translate)]
        rand_scale = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        rand_shear = [(random.random() - 1) * 2 * self.shear, (random.random() - 1) * 2 * self.shear]
        img = TF.affine(img, angle=0, translate=rand_translate, scale=rand_scale, shear=rand_shear)
        label = TF.affine(label, angle=0, translate=rand_translate, scale=rand_scale, shear=rand_shear)

        return img, label


# can add color change, very important!!!

# labeled image data augmentation
class RandomGenerator(object):
    def __init__(self):
        self.RandomVerticalFlip = RandomVerticalFlipWithLabel()
        self.RandomHorizontalFlip = RandomHorizontalFlipWithLabel()
        self.RandomRotate = RandomRotate()
        self.RandomAffine = RandomAffine()

    def __call__(self, image, label):
        label = label.unsqueeze(0)
        image, label = self.RandomHorizontalFlip(image, label)
        image, label = self.RandomVerticalFlip(image, label)
        image, label = self.RandomRotate(image, label)
        image, label = self.RandomAffine(image, label)
        # ...
        label = label.squeeze(0)
        
        return image, label


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



