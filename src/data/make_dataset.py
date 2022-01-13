# -*- coding: utf-8 -*-
import logging
import os
import sys
from os import path
from pathlib import Path

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
# @click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
@hydra.main(config_name= "makeDataset_conf.yaml" ,config_path="../../conf")
def main(cfg):
    # hydra.initialize(config_path="../../conf")
    # cfg= compose(config_name="makeDataset_conf", return_hydra_config=True, overrides=["hydra.runtime.cwd=."])
    """Runs data processing scripts to turn raw data from (input_filepath : ../raw)
    into cleaned data ready to be analyzed (saved in ../processed).
    """
    # logger = logging.getLogger(__name__)
    # logger.info("making final data set from raw data")
    # os.chdir(hydra.utils.get_original_cwd())
    # print("Working directory : {}".format(os.getcwd()))
    

    input_filepath = f"{cfg.hyperparameters.input_filepath}"
    output_filepath = f"{cfg.hyperparameters.output_filepath}"

    input_filepath = Path(input_filepath)

    # Check if path exists else raise error
    if not path.exists(input_filepath):
        raise ValueError("Input path does not exist")

    image_path = list(input_filepath.glob("**/*.png")) + list(input_filepath.glob("**/*.jpg"))
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

    # Saving in a DataFrame
    image_data = pd.DataFrame({'Path': non_segmented_images,\
              'labels': lables})
    ##########################
    ### FISH DATASET
    ##########################
    convert_tensor = transforms.ToTensor()

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


    for label in list(set(lables)):
        class_name = list(int_classes.keys())[list(int_classes.values()).index(label)]
        print(class_name)
        iter_num = 0
        dir_exist = os.path.exists(f'{output_filepath}{class_name}')
        if not dir_exist:
            os.mkdir(f'{output_filepath}{class_name}')
        for im in image_data[image_data.labels == label].Path:
            print(im)
            img = Image.open(im)
            img_tensor = convert_tensor(img)
            for i in range(10):
                out = aug_list(img_tensor)
                image = out[0].numpy().transpose((1,2,0))
                plt.imsave(f'{output_filepath}{class_name}\im{iter_num}.png', image)
                iter_num += 1 


    # Create Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        train,
        train_labels,
        val,
        val_labels,
        test,
        test_labels,
        cfg.hyperparameters.TRAIN_BATCHSIZE,
        1,
        train_transform,
        test_transforms,
    )

    print(train_loader)
    # Save data
    torch.save(train_loader, f"{output_filepath}train.pt")
    torch.save(val_loader, f"{output_filepath}test.pt")
    torch.save(test_loader, f"{output_filepath}val.pt")

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
    train_transform,
    test_transform,
):
    """
    Returns the Train, Validation and Test DataLoaders.
    """

    train_ds = FishDataset(images=train, labels=train_labels, transform=train_transform)
    val_ds = FishDataset(images=val, labels=val_labels, transform=test_transform)
    test_ds = FishDataset(images=test, labels=test_labels, transform=test_transform)

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

    

