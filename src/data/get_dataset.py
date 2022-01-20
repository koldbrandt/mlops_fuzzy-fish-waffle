# -*- coding: utf-8 -*-
from os import path
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from typing import List

from src.data.make_dataset import getImagesAndLabels


def get_params(cfg: OmegaConf):
    """
    Returns relevant parameters in config file
    """
    input_filepath = f"{cfg.paths.input_filepath}"
    input_filepath = Path(input_filepath)
    TRAIN_BATCHSiZE = cfg.hyperparameters.TRAIN_BATCHSIZE
    TEST_SIZE = cfg.hyperparameters.TEST_SIZE
    NUM_WORKERS = cfg.hyperparameters.NUM_WORKERS
    return input_filepath, TRAIN_BATCHSiZE, TEST_SIZE, NUM_WORKERS


def main(cfg: OmegaConf):
    """
    Runs data processing scripts to turn processed data from
    (input_filepath : ../processed)
    into dataloaders that will get returned.
    """
    # input_filepath = f"{cfg.hyperparameters.input_filepath}"
    # input_filepath = Path(input_filepath)
    input_filepath, TRAIN_BATCHSIZE, TEST_SIZE, NUM_WORKERS = get_params(cfg)

    # Check if path exists else raise error
    if not path.exists(input_filepath):
        raise ValueError("Input path does not exist")

    non_segmented_images, labels, _, _ = getImagesAndLabels(input_filepath)

    train, test, train_labels, test_labels = train_test_split(
        non_segmented_images, labels, test_size=TEST_SIZE, shuffle=True
    )

    train, val, train_labels, val_labels = train_test_split(
        train, train_labels, test_size=TEST_SIZE, shuffle=True
    )

    # FISH DATASET
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Create Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        train,
        train_labels,
        val,
        val_labels,
        test,
        test_labels,
        TRAIN_BATCHSIZE,
        NUM_WORKERS,
        transform,
    )

    return train_loader, val_loader, test_loader


class FishDataset(TensorDataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)
            label = self.labels[idx]
        return img, label


def get_loaders(
    train: Path,
    train_labels: List[int],
    val: Path,
    val_labels: List[int],
    test: Path,
    test_labels: List[int],
    batch_size: int,
    num_workers: int,
    transform, 
):
    """
    Returns the Train, Validation and Test DataLoaders.
    """

    train_ds = FishDataset(images=train, labels=train_labels, transform=transform)
    val_ds = FishDataset(images=val, labels=val_labels, transform=transform)
    test_ds = FishDataset(images=test, labels=test_labels, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return train_loader, val_loader, test_loader
