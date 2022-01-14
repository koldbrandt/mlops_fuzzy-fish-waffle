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


@hydra.main(config_name="makeDataset_conf.yaml", config_path="../../conf")
def main(cfg):
    """Runs data processing scripts to turn processed data from (input_filepath : ../processed)
    into dataloaders that will get returned. 
    """
    input_filepath = f"{cfg.hyperparameters.input_filepath}"
    output_filepath = f"{cfg.hyperparameters.output_filepath}"
    input_filepath = Path(input_filepath)

    # Check if path exists else raise error
    if not path.exists(input_filepath):
        raise ValueError("Input path does not exist")

    image_path = list(input_filepath.glob("**/*.png")) + list(
        input_filepath.glob("**/*.jpg")
    )
    # All path to images
    non_segmented_images = [img for img in image_path if "GT" not in str(img)]
    labels_non_segment = [img.parts[-2] for img in non_segmented_images]

    # All fish classes
    classes = list(set(labels_non_segment))
    print(f"Available Classes: {classes}")

    int_classes = {fish: i for i, fish in enumerate(classes)}
    lables = [int_classes[lable] for lable in labels_non_segment]

    # Label Dictionary
    print(int_classes)

    train, test, train_labels, test_labels = train_test_split(
        non_segmented_images, lables, test_size=0.2, shuffle=True
    )

    train, val, train_labels, val_labels = train_test_split(
        train, train_labels, test_size=0.2, shuffle=True
    )

    ##########################
    ### FISH DATASET
    ##########################

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
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
        cfg.TRAIN_BATCHSIZE,
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
        img = Image.open(self.images[idx])
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
