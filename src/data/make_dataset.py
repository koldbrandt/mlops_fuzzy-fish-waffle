# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from PIL import Image
from torchvision import transforms

from kornia.augmentation import ImageSequential
import kornia
import  matplotlib.pyplot as plt
import pandas as pd
import os 
import hydra

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
@hydra.main(config_name= "makeDataset_conf.yaml" ,config_path="../../conf")
def main(cfg):
    """Runs data processing scripts to turn raw data from (input_filepath : ../raw)
    into cleaned data ready to be analyzed (saved in ../processed).
    """
    # logger = logging.getLogger(__name__)
    # logger.info("making final data set from raw data")
    os.chdir(hydra.utils.get_original_cwd())
    print("Working directory : {}".format(os.getcwd()))
    # test_traindata_length()

    input_filepath = cfg.input_filepath
    output_filepath = cfg.output_filepath

    input_filepath = Path(input_filepath)

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

