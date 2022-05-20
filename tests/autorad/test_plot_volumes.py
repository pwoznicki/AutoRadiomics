from pathlib import Path

import numpy as np
import pytest

from autorad.config import config
from autorad.visualization.plot_volumes import BaseVolumes

data_dir = Path(config.TEST_DATA_DIR) / "nifti" / "prostate"

image_path = data_dir / "img.nii.gz"
mask_path = data_dir / "seg_multilabel.nii.gz"


class TestBaseVolumes:
    @pytest.mark.parametrize("constant_bbox", [True])
    def test_from_nifti(self, constant_bbox):
        self.plotter = BaseVolumes.from_nifti(
            image_path, mask_path, resample=True, constant_bbox=constant_bbox
        )
        assert self.plotter.image.shape == self.plotter.mask.shape
        assert np.sum(self.plotter.mask) > 0

    def test_crop_and_slice(self):
        # create a mask with zero-margin of 10 voxels
        mask = np.pad(
            np.ones((10, 10, 10)),
            pad_width=10,
            mode="constant",
        )
        return mask
