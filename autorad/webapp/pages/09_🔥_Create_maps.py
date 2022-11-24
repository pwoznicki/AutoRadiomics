from pathlib import Path

import streamlit as st

from autorad.config import config
from autorad.feature_extraction.voxelbased import extract_feature_maps
from autorad.utils import io
from autorad.webapp import template_utils, utils


def show():
    """Shows the sidebar components for the template
    and returns user inputs as dict."""
    input_dir = Path(utils.get_input_dir())
    result_dir = Path(utils.get_result_dir())
    with st.sidebar:
        if st.checkbox("Load test data to the input directory"):
            utils.load_test_data(input_dir)
    template_utils.find_all_data(input_dir)

    image_path, seg_path = template_utils.read_image_seg_paths()

    col1, _ = st.columns(2)
    with col1:
        output_dirname = st.text_input(
            "Give this extraction some ID to easily find the results:"
        )
        if not output_dirname:
            st.stop()
        else:
            maps_output_dir = result_dir / output_dirname
            if utils.dir_nonempty(maps_output_dir):
                st.warning("This ID already exists and has some data!")
            else:
                maps_output_dir.mkdir(parents=True, exist_ok=True)
                st.success(f"Maps will be saved in {maps_output_dir}")

    extraction_params = template_utils.radiomics_params_voxelbased()

    start_extraction = st.button("Get feature maps!")
    if start_extraction:
        assert output_dirname, "You need to assign an ID first! (see above)"
        assert image_path, "You need to provide an image path!"
        assert seg_path, "You need to provide a segmentation path!"
        io.save_yaml(
            extraction_params, maps_output_dir / "extraction_params.yaml"
        )
        with st.spinner("Extracting and saving feature maps..."):
            extract_feature_maps(
                image_path, seg_path, str(maps_output_dir), extraction_params
            )
        st.success(
            f"Done! Feature maps and configuration saved in {maps_output_dir}"
        )
        if config.IS_DEMO:
            zip_name = f"{output_dirname}.zip"
            zip_save_path = result_dir / zip_name
            utils.zip_directory(str(maps_output_dir), str(zip_save_path))
            with open(str(zip_save_path), "rb") as fp:
                st.download_button(
                    "Download results",
                    data=fp,
                    file_name=zip_name,
                    mime="application/zip",
                )


if __name__ == "__main__":
    show()
