from pathlib import Path

import pandas as pd
import streamlit as st

from autorad.utils import preprocessing


def show():
    st.write(
        """
    #####
    Before you proceed with the next tasks you need to create a .csv table.
    The table should contain the following information for each case:
    - **ID** (has to be unique)
    - Path to the **image**
    - Path to the **mask**
    - **Label** (0 or 1)

    Example:
    """
    )
    example_path_df = pd.read_csv("./webapp/paths_example.csv")
    st.write(example_path_df)
    st.write(
        """
    #####
    You can create the table manually (e.g. in Excel) or, if your dataset
    follows the specified structure, you can generate it below.
    """
    )
    dir_structures = {
        "Separate folder for each case": """
            Expected format:
            ```
            <dataset folder>
            +-- case1
            |   +-- image.nii.gz
            |   +-- segmentation.nii.gz
            +-- case2
            |   +-- image.nii.gz
            |   +-- segmentation.nii.gz
            ...
            ```
            """,
        "Separate folder for each case with multiple segmentations per image": """
            Expected format:
            ```
            <dataset folder>
            +-- case1
            |   +-- image.nii.gz
            |   +-- segmentation1.nii.gz
            |   +-- segmentation002.nii.gz
            |   +-- segmentation_03.nii.gz
            ...
            +-- case2
            |   +-- image.nii.gz
            |   +-- segmentation_first.nii.gz
            |   +-- segmentation_second.nii.gz
            ...
            ```
            """,
        "One folder for all the images, and one for the segmentations": """
            Expected format:
            ```
            <dataset folder>
            +-- images
            |   +-- 1.nii.gz
            |   +-- 2.nii.gz
            |   +-- ...
            +-- segmentations
            |   +-- 1.nii.gz
            |   +-- 2.nii.gz
            |   +-- ...
            ...
            ```
            """,
    }
    formats = list(dir_structures.keys())[:2]  # no support for 3rd option
    col1, col2 = st.columns(2)
    with col1:
        format = st.selectbox("Dataset structure", formats)
    with col2:
        st.write(dir_structures[format])
    data_dir = st.text_input(
        "Enter absolute path to the folder with your dataset:"
    )
    if not data_dir:
        st.stop()
    data_dir = Path(data_dir).absolute()
    if not data_dir.is_dir():
        st.error(f"The entered path is not a directory ({data_dir})")
    else:
        st.success(f"The entered path is a directory! ({data_dir})")
        col1, col2 = st.columns(2)
        with col1:
            image_stem = st.text_input(
                "How to identify image files? What root does every filename contain?",
                value="image",
            )
        with col2:
            mask_stem = st.text_input(
                "How to identify segmentation files? What root does every filename contain?",
                value="segmentation",
            )
        if format == formats[0]:
            paths_df = preprocessing.get_paths_with_separate_folder_per_case(
                data_dir=data_dir,
                image_stem=image_stem,
                mask_stem=mask_stem,
            )
        else:
            paths_df = (
                preprocessing.get_paths_with_separate_folder_per_case_loose(
                    data_dir=data_dir,
                    image_stem=image_stem,
                    mask_stem=mask_stem,
                )
            )

        st.download_button(
            "Download table ⬇️", paths_df.to_csv(), file_name="paths.csv"
        )
        st.dataframe(paths_df)


if __name__ == "__main__":
    show()
