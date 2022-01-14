# -*- coding: utf-8 -*-
import logging
import os
import sys
from os import path
from pathlib import Path

from kornia.augmentation import ImageSequential
import  matplotlib.pyplot as plt
import pandas as pd

import hydra
import pytest
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
import kornia

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


# from tests.test_data import test_traindata_length
def getImagesAndLabels(input_filepath: Path):
    """
    Takes a path as input and returns all images(PNG and JPG) in path 
    """
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

    uniqlabels = list(set(lables))

    return non_segmented_images, lables, uniqlabels, int_classes

def get_params(cfg):
    """
    Returns all parameters in config file
    """
    input_filepath = f"{cfg.paths.input_filepath}"
    output_filepath = f"{cfg.paths.output_filepath}"
    input_filepath = Path(input_filepath)

    return input_filepath, output_filepath



@hydra.main(config_name="dataset_conf.yaml", config_path="../../conf")
def main(cfg):
    """Runs data processing scripts to turn raw data from (input_filepath : ../raw)
    into cleaned data ready to be analyzed (saved in ../processed).
    """

    input_filepath, output_filepath = get_params(cfg)

    # Check if path exists else raise error
    if not path.exists(input_filepath):
        raise ValueError("Input path does not exist")
    if not path.exists(output_filepath):
        raise ValueError("Output path does not exist")

    
    non_segmented_images, labels, uniqLabels, int_classes = getImagesAndLabels(input_filepath)

    # Saving in a DataFrame
    image_data = pd.DataFrame({"Path": non_segmented_images, "labels": labels})
    ##########################
    ### FISH DATASET
    ##########################
    convert_tensor = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor()])

    aug_list = ImageSequential(
        #     kornia.color.BgrToRgb(),
        kornia.augmentation.ColorJitter(0.2, 0.0, 0.0, 0.0, p=1.0),
        #     kornia.filters.MedianBlur((3, 3)),
        kornia.augmentation.RandomAffine(360, p=1.0),
        #     kornia.augmentation.RandomGaussianNoise(mean=0., std=1., p=0.5),
        kornia.augmentation.RandomPerspective(0.1, p=0.8),
        kornia.augmentation.RandomHorizontalFlip(p=0.5)
        #     kornia.enhance.Invert(),
        #     kornia.augmentation.RandomMixUp(p=1.0),
        #     return_transform=True,
        #     same_on_batch=True
        #     random_apply=10
    )


    for label in uniqLabels:
        class_name = list(int_classes.keys())[list(int_classes.values()).index(label)]
        print(class_name)
        dir_exist = os.path.exists(f"{output_filepath}{class_name}")
        if not dir_exist:
            os.mkdir(f"{output_filepath}{class_name}")
        counter = 0
        for im in image_data[image_data.labels == label].Path:
            print(im)
            img = Image.open(im)
            img_tensor = convert_tensor(img).unsqueeze(0).repeat(10,1,1,1)
            out = aug_list(img_tensor)
            for i in range(10):
                image = out[i].numpy().transpose((1, 2, 0))
                plt.imsave(f"{output_filepath}{class_name}\im{counter}{i}.png", image)
            counter += 1


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
