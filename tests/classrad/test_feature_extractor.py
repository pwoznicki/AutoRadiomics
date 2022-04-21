import pytest

from autorad.data.dataset import ImageDataset
from autorad.feature_extraction.extractor import FeatureExtractor


@pytest.fixture
def small_image_dataset(small_paths_df):
    image_dataset = ImageDataset(
        df=small_paths_df,
        image_colname="img",
        mask_colname="seg",
        ID_colname="ID",
    )
    return image_dataset


def test_get_features(small_image_dataset, helpers):
    extractor = FeatureExtractor(
        dataset=small_image_dataset, out_path=helpers.mkdtemp(), verbose=False
    )
    return extractor


# def test_get_feature_names():
#     extractor = FeatureExtractor()
#     feature_names = extractor.get_feature_names()
#     return feature_names
