import os
import tempfile
from pathlib import Path

import pytest

from autorad.config import config
from autorad.utils import conversion


@pytest.mark.skip(reason="requires external dependency")
def test_dicom_to_nifti():
    dicom_dir = Path(config.TEST_DATA_DIR) / "DICOM"
    with tempfile.TemporaryDirectory() as tmp_dir:
        conversion.dicom_to_nifti(dicom_dir, tmp_dir)
        assert len(os.listdir(tmp_dir)) == 2
