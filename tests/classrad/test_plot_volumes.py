from pathlib import Path

import nibabel as nib
import numpy as np

from classrad.visualization.plot_volumes import BaseVolumes

data_dir = Path(__file__).parent.parent / "testing_data" / "nifti"

image_path = data_dir / "img.nii.gz"
mask_path = data_dir / "seg_multilabel.nii.gz"

image = nib.load(image_path).get_fdata()
mask = nib.load(mask_path).get_fdata()


class TestBaseVolumePlotter:
    def test_from_nifti(self):
        plotter = BaseVolumes.from_nifti(image_path, mask_path)
        assert np.array_equal(plotter.image, image)
        assert np.array_equal(plotter.mask, mask)
