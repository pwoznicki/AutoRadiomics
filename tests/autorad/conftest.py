import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import settings

from autorad.config import config
from autorad.data.dataset import FeatureDataset

settings.register_profile("fast", max_examples=2)
settings.register_profile("slow", max_examples=10)


@pytest.fixture
def small_paths_df():
    data_dir = Path(config.TEST_DATA_DIR) / "nifti" / "prostate"
    img_path = data_dir / "img.nii.gz"
    one_label_mask_path = data_dir / "seg_one_label.nii.gz"
    multi_label_mask_path = data_dir / "seg_two_labels.nii.gz"
    paths_df = pd.DataFrame(
        {
            "ID": [0, 1],
            "img": [img_path, img_path],
            "seg": [one_label_mask_path, multi_label_mask_path],
        }
    )
    return paths_df


@pytest.fixture
def empty_df():
    df = pd.DataFrame()
    return df


@pytest.fixture
def binary_df():
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "Feature1": [0.0 for i in range(100)],
            "Label": [i % 2 for i in range(100)],
        }
    )


@pytest.fixture
def multiclass_df():
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "Feature1": [100 * i for i in range(100)],
            "Label": [i % 4 for i in range(100)],
        }
    )


@pytest.fixture
def feature_dataset(df):
    return FeatureDataset(
        dataframe=df,
        ID_colname="ID",
        target="Label",
    )


class Helpers:
    """For common utils, following advice from
    https://stackoverflow.com/questions/33508060
    """

    @staticmethod
    def tmp_dir():
        dirpath = tempfile.mkdtemp()
        return Path(dirpath)


@pytest.fixture
def helpers():
    return Helpers
