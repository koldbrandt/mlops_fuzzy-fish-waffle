# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset
import hydra
import os
import sys
from os import path
from src.data.make_dataset import getImagesAndLabels


def get_params(cfg):
    """
    Returns relevant parameters in config file
    """
    input_filepath = f"{cfg.paths.input_filepath}"
    input_filepath = Path(input_filepath)
    TRAIN_BATCHSiZE = cfg.hyperparameters.TRAIN_BATCHSIZE
    TEST_SIZE = cfg.hyperparameters.TEST_SIZE

    return input_filepath, TRAIN_BATCHSiZE, TEST_SIZE


# @hydra.main(config_name="dataset_conf.yaml", config_path="../../conf")
def main(cfg):
    """
    Runs data processing scripts to turn processed data from (input_filepath : ../processed)
    into dataloaders that will get returned. 
    """
    # input_filepath = f"{cfg.hyperparameters.input_filepath}"
    # input_filepath = Path(input_filepath)
    input_filepath, TRAIN_BATCHSIZE, TEST_SIZE  = get_params(cfg)

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

    ##########################
    ### FISH DATASET
    ##########################

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
        1,
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
        img = Image.open(self.images[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)
            label = self.labels[idx]
        return img, label


def get_loaders(
    train,
    train_labels,
    val,
    val_labels,
    test,
    test_labels,
    batch_size,
    num_workers,
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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # pytest.main(["-qq"], plugins=[FishDataset()])
    main()
