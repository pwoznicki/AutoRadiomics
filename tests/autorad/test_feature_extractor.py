import pytest
from conftest import prostate_data

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
    print("blabla")


@pytest.mark.skip(reason="needs first to provide new API for extractor")
def test_get_features_for_single_case(feature_extractor):
    image_path = prostate_data["img"]
    mask_path = prostate_data["seg"]
    feature_dict = feature_extractor.get_features_for_single_case(
        image_path=image_path,
        mask_path=mask_path,
    )
    assert all(isinstance(val, float) for val in feature_dict.values())

    empty_mask_path = prostate_data["empty_seg"]
    with pytest.raises(ValueError):
        _ = feature_extractor.get_features_for_single_case(
            image_path=image_path,
            mask_path=empty_mask_path,
        )


def test_get_pyradiomics_feature_names(feature_extractor):
    feature_names = feature_extractor.get_pyradiomics_feature_names()
    print(feature_names)
