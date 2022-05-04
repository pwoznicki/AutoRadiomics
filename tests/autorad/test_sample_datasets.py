import os

import pytest
from monai.apps.datasets import DecathlonDataset

from autorad.config import config
from autorad.utils.sample_datasets import (
    convert_decathlon_dataset,
    load_mednist_dataset,
)


@pytest.mark.skip(reason="Slow")
def test_convert_decathlon_dataset():
    decathlon_dataset = DecathlonDataset(
        root_dir=config.MONAI_DATA_DIR,
        task="Task05_Prostate",
        section="training",
        download=True,
    )
    autorad_dataset = convert_decathlon_dataset(decathlon_dataset)
    df = autorad_dataset.dataframe()
    assert len(df) > 0
    images = autorad_dataset.image_paths()
    assert os.path.exists(images[0])
    masks = autorad_dataset.mask_paths()
    assert os.path.exists(masks[0])
    ids = autorad_dataset.ids()
    assert any([isinstance(ids[0], str), isinstance(ids[0], int)])


@pytest.mark.skip(reason="Slow")
def test_load_mednist_dataset(helpers):
    tmp_dir = helpers().tmp_dir()
    image_dataset = load_mednist_dataset(tmp_dir)
    assert len(image_dataset.dataframe()) > 0
    image_paths = image_dataset.image_paths()
    mask_paths = image_dataset.mask_paths()
    ids = image_dataset.ids()
    assert type(ids[0]) == str
    assert os.path.exists(image_paths[0])
    assert os.path.exists(mask_paths[0])
