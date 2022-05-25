import pytest

from autorad.data.dataset import ImageDataset
from autorad.feature_extraction.extractor import FeatureExtractor


@pytest.fixture
def image_dataset(small_paths_df):
    image_dataset = ImageDataset(
        df=small_paths_df,
        image_colname="img",
        mask_colname="seg",
        ID_colname="ID",
    )
    return image_dataset


@pytest.fixture
def feature_extractor(image_dataset):
    feature_extractor = FeatureExtractor(
        dataset=image_dataset,
    )
    return feature_extractor


def test_run(feature_extractor):
    result_df = feature_extractor.run()
    assert len(result_df) == 2


def test_get_features_for_single_case():
    pass


def test_get_pyradiomics_feature_names(feature_extractor):
    feature_names = feature_extractor.get_pyradiomics_feature_names()
    print(feature_names)
