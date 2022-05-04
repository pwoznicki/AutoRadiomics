from pathlib import Path

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    SaveImaged,
)
from radiomics import featureextractor

from autorad.config import config

if __name__ == "__main__":
    data_dir = Path(config.TEST_DATA_DIR)
    img_path = data_dir / "DICOM" / "CT_head"
    mask_path = data_dir / "nifti" / "CT_head_seg.nii.gz"
    input = {"img": img_path, "seg": mask_path}
    out_dir = Path(data_dir / "out")
    pipeline = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["img", "seg"]),
            SaveImaged(
                keys=["img", "seg"],
                meta_keys=["img_meta_dict", "seg_meta_dict"],
                output_dir=out_dir,
                separate_folder=False,
            ),
        ]
    )
    pipeline(input)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    trans_img_path = out_dir / "CT_head_trans.nii.gz"
    trans_seg_path = out_dir / "CT_head_seg_trans.nii.gz"
    result = extractor.execute(str(trans_img_path), str(trans_seg_path))
    print(result)
