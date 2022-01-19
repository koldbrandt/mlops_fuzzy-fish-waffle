# If data is not present then skip
import os

import pytest
import torch
from omegaconf import OmegaConf

# import src.data.make_dataset
from src.data.get_dataset import main


# import _src.__main__
@pytest.mark.skipif(
    not os.path.exists("data/processed/"), reason="Train files not found"
)
@pytest.mark.skipif(
    not os.path.exists("data/processed/"), reason="Test files not found"
)

# See how much of data is run based on tests
# RUN test : coverage run -m pytest tests/
# See coverage : coverage report


class TestClass:
    """Tests on get_dataset"""

    def test_InputPath_Exists(self):
        """
        Test value error raised if input_path does not exist
        """
        CONFIG = OmegaConf.create(
            {
                "hyperparameters": {
                    "TRAIN_BATCHSIZE": 4,
                    "TEST_SIZE": 0.2,
                },
                "paths": {
                    "input_filepath": os.getcwd() + "/test/some/path",
                },
            }
        )
        with pytest.raises(ValueError, match="Input path does not exist"):
            main(CONFIG)

    def test_outputImgSize(self):
        """
        Test if images from get_dataset are 64x64
        """
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
        train_loader, _, test_loader = main(CONFIG)
        for images, _ in train_loader:
            assert images.shape[2:4] == torch.Size([64, 64])

        for images, _ in test_loader:
            assert images.shape[2:4] == torch.Size([64, 64])
