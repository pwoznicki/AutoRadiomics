from pathlib import Path

import SimpleITK as sitk
import six
import streamlit as st
import utils
from radiomics import featureextractor

result_dir = Path("/Users/p.woznicki/Documents/test")


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""
    col1, col2 = st.columns(2)
    with col1:
        upload = utils.upload_file("Upload the image:")
        image = sitk.ReadImage(upload)
    #      image =
    with col2:
        upload = utils.upload_file("Upload the mask:")
        mask = sitk.ReadImage(upload)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    feature_vector = extractor.execute(image, mask, voxelBased=True)
    for feature_name, feature_value in feature_vector.items():
        save_path = result_dir / f"{feature_name}.nrrd"
        sitk.WriteImage(feature_value, save_path)


if __name__ == "__main__":
    show()
