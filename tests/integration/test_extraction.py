from autorad.data import ImageDataset
from autorad.feature_extraction import FeatureExtractor
import pandas as pd
import numpy as np
from pathlib import Path

from autorad.config import config

def test_extraction_from_dicom():
    base_dir = Path(config.TEST_DATA_DIR) / "DICOM" / "Prostate-MRI-US-Biopsy-001"
    path_df = pd.read_csv(base_dir / "paths.csv")
    dataset = ImageDataset(
        df=path_df,
        ID_colname="ID",
        image_colname="img_path",
        mask_colname="seg_path",
        root_dir=base_dir,
    )
    extractor = FeatureExtractor(
        dataset, extraction_params="MR_default.yaml", n_jobs=-1
    )
    feature_df = extractor.run()
    for col in feature_df.columns:
        first_val = feature_df.iloc[0][col]
        second_val = feature_df.iloc[1][col]
        if type(first_val) is not str:
            assert np.isclose(first_val, second_val, rtol=1e-2)


def test_extraction_for_various_labels():
    base_dir = Path(config.TEST_DATA_DIR) / "DICOM" / "UPENN-GBM-00002"
    path_df = pd.read_csv(base_dir / "paths.csv")
    dataset = ImageDataset(
        df=path_df,
        ID_colname="ID",
        image_colname="img_path",
        mask_colname="seg_path",
        root_dir=base_dir,
    )
    extractor = FeatureExtractor(
        dataset, extraction_params="MR_default.yaml", n_jobs=-1
    )
    feature_df = {}
    for label in [1, 2, 4]:
        feature_df[label] = extractor.run(mask_label=label)
        assert len(feature_df[label]) == 1


def test_extraction_for_dicom_seg():
    base_dir = Path(config.TEST_DATA_DIR) / "DICOM" / "QIN-PROSTATE-001"
    path_df = pd.read_csv(base_dir / "paths.csv")
    dataset = ImageDataset(
        df=path_df,
        ID_colname="ID",
        image_colname="img_path",
        mask_colname="seg_path",
        root_dir=base_dir,
    )
    extractor = FeatureExtractor(
        dataset, extraction_params="MR_default.yaml", n_jobs=-1,
    )
    feature_df = extractor.run(mask_label=1)
    assert len(feature_df) == 1