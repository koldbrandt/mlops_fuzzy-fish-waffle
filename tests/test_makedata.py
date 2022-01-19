# If data is not present then skip
import os
import os.path
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# import src.data.make_dataset
from src.data.make_dataset import getImagesAndLabels, main


# import _src.__main__
@pytest.mark.skipif(not os.path.exists("data/raw/"), reason="Train files not found")
@pytest.mark.skipif(
    not os.path.exists("data/processed/"), reason="Test files not found"
)

# See how much of data is run based on tests
# RUN test : coverage run -m pytest tests/
# See coverage : coverage report


class TestClass:
    """
    Tests on make_dataset
    """

    def test_InputPath_Exists(self):
        """
        Test value error raised if input_path does not exist
        """
        CONFIG = OmegaConf.create(
            {
                "hyperparameters": {
                    "TRAIN_BATCHSIZE": 64,
                },
                "paths": {
                    "input_filepath": os.getcwd() + "/test/some/path",
                    "output_filepath": os.getcwd() + "/data/processed",
                },
            }
        )

        with pytest.raises(ValueError, match="Input path does not exist"):
            main(CONFIG)

    def test_OutputPath_Exists(self):
        """
        Test value error raised if output_path does not exist
        """
        CONFIG = OmegaConf.create(
            {
                "hyperparameters": {
                    "TRAIN_BATCHSIZE": 64,
                },
                "paths": {
                    "input_filepath": os.getcwd() + "/data/raw/",
                    "output_filepath": os.getcwd() + "/test/some/path",
                },
            }
        )

        with pytest.raises(ValueError, match="Output path does not exist"):
            main(CONFIG)

    def test_allLabelsRepresented(self):
        _, _, uniqueLabels, _ = getImagesAndLabels(Path(os.getcwd() + "/data/raw/"))
        print(uniqueLabels)
        assert uniqueLabels == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def test_shapeOfImages(self):
        # 64 x 64?
        assert True
