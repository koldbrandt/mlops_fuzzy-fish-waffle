# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Runs data processing scripts to turn raw data from (input_filepath : ../raw) 
    into cleaned data ready to be analyzed (saved in ../processed).

    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_filepath = Path(input_filepath)

    image_path = list(input_filepath.glob("**/*.png"))

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

    # Create data
    image_data = pd.DataFrame({"Path": non_segmented_images, "labels": lables})
   
   

    train,test, train_labels, test_labels = train_test_split(image_data.Path, image_data.labels, test_size=0.2, shuffle=True)
    
    # Create data
    train_data = pd.DataFrame({"Path": train, "labels": train_labels})
    test_data = pd.DataFrame({"Path": test, "labels": test_labels})

    # Save data
    train_data.to_pickle(f"{output_filepath}train.pkl")
    test_data.to_pickle(f"{output_filepath}test.pkl")


    # train,val, train_labels, val_labels = train_test_split(train, train_labels, test_size=0.2, shuffle=True)

    



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
