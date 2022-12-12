import logging
import shutil
from pathlib import Path

import radiomics
import SimpleITK as sitk

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.utils import io


def extract_feature_maps(
    image_path: PathLike,
    seg_path: PathLike,
    save_dir: PathLike,
    extraction_params: dict | None = None,
    copy_inputs: bool = True,
):
    save_dir = Path(save_dir)
    if copy_inputs:
        shutil.copyfile(image_path, save_dir / "image.nii.gz")
        shutil.copyfile(seg_path, save_dir / "segmentation.nii.gz")
    if extraction_params is None:
        extraction_params = io.load_yaml(
            Path(config.PARAM_DIR) / "CT_default_feature_map.yaml"
        )
    radiomics.setVerbosity(logging.INFO)
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
        extraction_params
    )
    feature_vector = extractor.execute(
        str(image_path), str(seg_path), voxelBased=True
    )
    for feature_name, feature_value in feature_vector.items():
        if isinstance(feature_value, sitk.Image):
            save_path = save_dir / f"{feature_name}.nii.gz"
            save_path = str(save_path)
            sitk.WriteImage(feature_value, save_path)
