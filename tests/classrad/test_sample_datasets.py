from classrad.utils.sample_datasets import (
    convert_decathlon_dataset,
)
from classrad.config import config
from monai.apps.datasets import DecathlonDataset
import os
import pytest


@pytest.mark.slow
def test_convert_decathlon_dataset():
    decathlon_dataset = DecathlonDataset(
        root_dir=config.MONAI_DATA_DIR,
        task="Task05_Prostate",
        section="training",
        download=True,
    )
    classrad_dataset = convert_decathlon_dataset(decathlon_dataset)
    df = classrad_dataset.dataframe()
    assert len(df) > 0
    images = classrad_dataset.image_paths()
    assert os.path.exists(images[0])
    masks = classrad_dataset.mask_paths()
    assert os.path.exists(masks[0])
    ids = classrad_dataset.ids()
    assert any([isinstance(ids[0], str), isinstance(ids[0], int)])
