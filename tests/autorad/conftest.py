import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import settings

from autorad.config import config
from autorad.data.dataset import FeatureDataset

settings.register_profile("fast", max_examples=2)
settings.register_profile("slow", max_examples=10)

prostate_root = Path(config.TEST_DATA_DIR) / "nifti" / "prostate"
prostate_data = {
    "img": prostate_root / "img.nii.gz",
    "seg": prostate_root / "seg_one_label.nii.gz",
    "seg_two_labels": prostate_root / "seg_two_labels.nii.gz",
    "empty_seg": prostate_root / "seg_empty.nii.gz",
}


@pytest.fixture
def small_paths_df():
    paths_df = pd.DataFrame(
        {
            "ID": ["case_1_single_label", "case_2_two_labels"],
            "img": [prostate_data["img"], prostate_data["img"]],
            "seg": [prostate_data["seg"], prostate_data["seg_two_labels"]],
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


def feature_df():
    return pd.DataFrame(
        {
            "ID": [str(i) for i in range(100)],
            "Feature1": [100 * i for i in range(100)],
            "Feature2": [i % 2 for i in range(100)],
            "Label": [i % 2 for i in range(100)],
        }
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
