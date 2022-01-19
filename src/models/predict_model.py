import logging
import os

import click
import model as md
import torch
from dotenv import find_dotenv, load_dotenv
from model import Network
from omegaconf import OmegaConf
from torch import nn

import src.data.get_dataset


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
def main(
    model_filepath: str = "models/checkpoint.pth",
):
    CONFIG = OmegaConf.create(
        {
            "hyperparameters": {
                "TRAIN_BATCHSIZE": 64,
                "TEST_SIZE": 0.2,
            },
            "paths": {
                "input_filepath": os.getcwd() + "/data/processed/",
            },
        }
    )
    model = load_checkpoint(model_filepath)
    _, valloader, _ = src.data.get_dataset.main(CONFIG)
    criterion = nn.CrossEntropyLoss()

    test_loss, accuracy = md.validation(model, valloader, criterion)
    print(
        "Test Loss: {:.3f}.. ".format(test_loss / len(valloader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(valloader)),
    )


def load_checkpoint(model_filepath: str):
    """
    Loads saved model and returns it
    """
    checkpoint = torch.load(model_filepath)
    saved_model = Network(9)
    saved_model.load_state_dict(checkpoint["state_dict"])

    return saved_model


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
