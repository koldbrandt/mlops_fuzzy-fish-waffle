import os
import os.path

import pytest
import torch
from omegaconf import OmegaConf

import src.data.get_dataset
from src.models.model import Network


@pytest.mark.skipif(
    not os.path.exists("data/processed/Trout"), reason="Train files not found"
)
def test_inputshape_has_outputshape():
    CONFIG = OmegaConf.create(
        {
            "hyperparameters": {
                "epochs": 10,
                "print_every": 40,
                "lr": 0.01,
                "momentum": 0.9,
                "num_classes": 9,
                "TRAIN_BATCHSIZE": 64,
                "TEST_SIZE": 0.2,
            },
            "paths": {"input_filepath": os.getcwd() + "/data/processed/"},
        }
    )
    model = Network(CONFIG.hyperparameters.num_classes)
    trainloader, _, testloader = src.data.get_dataset.main(CONFIG)
    model.train()
    for images, labels in trainloader:
        output = model(images)
        assert (
            images.shape[2:4] == torch.Size([64, 64]) and output.shape[1] == 9
        ), "Either images did not have shape 64x64 or the model output shape is not 9"


# Test raise error
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected each image to have shape 64 x 64"):
        model = Network(9)  # My model
        model(torch.randn(1, 1, 4, 4))
