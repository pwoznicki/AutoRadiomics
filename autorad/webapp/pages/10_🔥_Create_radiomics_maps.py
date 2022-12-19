from pathlib import Path

import streamlit as st

from autorad.config import config
from autorad.feature_extraction.voxelbased import extract_feature_maps
from autorad.utils import io
from autorad.webapp import extraction_utils, st_read, st_utils


def show():
    st_utils.show_title()
    input_dir = Path(st_read.get_input_dir())
    result_dir = Path(st_read.get_result_dir())
    st_read.find_all_data(input_dir)

    image_path, seg_path = st_read.read_image_seg_paths()

    col1, _ = st.columns(2)
    with col1:
        output_dirname = st.text_input(
            "Give this extraction some ID to easily find the results:"
        )
        if not output_dirname:
            st.stop()
        else:
            maps_output_dir = result_dir / output_dirname
            if st_read.dir_nonempty(maps_output_dir):
                st.warning("This ID already exists and has some data!")
            else:
                maps_output_dir.mkdir(parents=True, exist_ok=True)
                st.success(f"Maps will be saved in {maps_output_dir}")

    extraction_params = extraction_utils.radiomics_params_voxelbased()

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
            io.zip_directory(str(maps_output_dir), str(zip_save_path))
            with open(str(zip_save_path), "rb") as fp:
                st.download_button(
                    "Download results",
                    data=fp,
                    file_name=zip_name,
                    mime="application/zip",
                )


if __name__ == "__main__":
    show()
