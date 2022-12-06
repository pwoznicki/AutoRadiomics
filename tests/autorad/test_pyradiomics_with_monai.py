import tempfile
from pathlib import Path

from radiomics import featureextractor

from autorad.config import config
from autorad.utils import conversion

if __name__ == "__main__":
    data_dir = Path(config.TEST_DATA_DIR)
    img_path = data_dir / "DICOM" / "CT_head"
    seg_path = data_dir / "nifti" / "CT_head_seg.nii.gz"
    out_path = Path(data_dir / "out")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_img_path = Path(tmp_dir) / "CT_head_trans.nii.gz"
        conversion.convert_to_nifti(img_path, out_img_path)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute(str(out_img_path), str(seg_path))
        print(result)
