from pathlib import Path
import os
import SimpleITK as sitk
import streamlit as st
import utils
from radiomics import featureextractor
from template_utils import radiomics_params_voxelbased

import radiomics
import logging

input_dir = Path(os.environ["INPUT_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""
    with st.sidebar:
        load_test_data = st.checkbox("Load test data to input directory")
    if load_test_data:
        utils.load_test_data(input_dir)
    filelist = []
    # Find all files in input directory ending with '.nii.gz' or '.nrrd'
    for fpath in input_dir.rglob("*.nii.gz"):
        filelist.append(str(fpath))
    for fpath in input_dir.rglob("*.nii"):
        filelist.append(str(fpath))
    for fpath in input_dir.rglob("*.nrrd"):
        filelist.append(str(fpath))
    filelist = [fpath for fpath in filelist if not "results" in fpath]
    st.write("""Files found in your input directory:""")
    st.write(filelist)

    col1, col2 = st.columns(2)
    with col1:
        image_path = st.text_input("Paste here the path to the image:")
        image_path = image_path.strip('"')
        if os.path.isfile(image_path):
            st.success("Image found!")
    with col2:
        seg_path = st.text_input("Paste here the path to the segmentation:")
        seg_path = seg_path.strip('"')
        if os.path.isfile(seg_path):
            st.success("Segmentation found!")
    col1, col2 = st.columns(2)
    with col1:
        output_dirname = st.text_input(
            "Give this extraction some ID to easily find the results:"
        )
        if output_dirname:
            maps_output_dir = result_dir / output_dirname
            if utils.dir_nonempty(maps_output_dir):
                st.warning("This ID already exists and has some data!")
            else:
                maps_output_dir.mkdir(parents=True, exist_ok=True)
                st.success(f"Maps will be saved in {maps_output_dir}")
    extraction_params = radiomics_params_voxelbased()
    start_extraction = st.button("Get feature maps!")
    if start_extraction:
        assert output_dirname, "You need to assign an ID first! (see above)"
        assert image_path, "You need to provide an image path!"
        assert seg_path, "You need to provide a segmentation path!"
        utils.save_yaml(extraction_params, maps_output_dir / "extraction_params.yaml")
        with st.spinner("Extracting and saving feature maps..."):
            radiomics.setVerbosity(logging.INFO)
            extractor = featureextractor.RadiomicsFeatureExtractor(extraction_params)
            feature_vector = extractor.execute(image_path, seg_path, voxelBased=True)
            for feature_name, feature_value in feature_vector.items():
                if isinstance(feature_value, sitk.Image):
                    save_path = maps_output_dir / f"{feature_name}.nii.gz"
                    save_path = str(save_path)
                    sitk.WriteImage(feature_value, save_path)
        st.success(f"Done! Feature maps and configuration saved in {maps_output_dir}")


if __name__ == "__main__":
    show()
