import pathlib
import click
import logging
from dotenv import find_dotenv, load_dotenv
import torch
from torch import nn
from model import Network
import model as md
from src.data.make_dataset import FishDataset


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("predict_filepath", type=click.Path())
def main(model_filepath: str = "models/checkpoint.pth", predict_filepath: str = "data/processed/val.pt"):
    model = load_checkpoint(model_filepath)
    valloader = torch.load(predict_filepath)
    criterion = nn.CrossEntropyLoss()

    test_loss, accuracy = md.validation(model, valloader, criterion)
    print(
            "Test Loss: {:.3f}.. ".format(test_loss / len(valloader)),
            "Test Accuracy: {:.3f}".format(accuracy / len(valloader)),
        )

def load_checkpoint(model_filepath:str):     
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