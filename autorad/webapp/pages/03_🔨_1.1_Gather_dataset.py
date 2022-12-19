from pathlib import Path

import pandas as pd
import streamlit as st

from autorad.external.download_WORC import download_WORCDatabase
from autorad.utils import preprocessing
from autorad.webapp import st_read, st_utils, validation_utils


def download_example_dataset(input_dir):
    with st.expander("You have no data? Then download an example dataset 🫁"):
        st.write(
            """
            You can download the Desmoid dataset from [WORC](https://github.com/MStarmans91/WORCDatabase).

            It contains MRI images and segmentations of soft-tissue tumors classified as either fibromatosis (benign) or sarcomas (malignant).
            We can use this dataset to extract features and train a model that can distinguish between the two.
            """
        )
        col1, col2 = st.columns(2)
        with col1:
            n_subjects = st.slider(
                "Number of cases",
                min_value=5,
                max_value=100,
                value=10,
                help="If you want to train a model, it's best to download 100 cases.",
            )
        with col2:
            dataset_dir = st.text_input(
                "Where to save the example dataset",
                value=input_dir,
            )
        worc_citation = """
        If you use this dataset in your publication, please cite:

        Starmans, M. P. A. et al. (2021). The WORC* database: MRI and CT scans, segmentations, and clinical labels for 932 patients from six radiomics studies.

        Starmans, M. P. A. et al. (2021). Reproducible radiomics through automated machine learning validated on twelve clinical applications.
        """
        if st.button("Download example dataset", help=worc_citation):
            with st.spinner("Downloading dataset..."):
                download_WORCDatabase(
                    dataset="Desmoid",
                    data_folder=dataset_dir,
                    n_subjects=n_subjects,
                )
            st.success(
                f"Downloaded example dataset to {dataset_dir}.\n"
                f"You can use this path to generate the table with paths below."
            )


def show():
    st_utils.show_title()
    input_dir = st_read.get_input_dir()
    result_dir = st_read.get_result_dir()
    download_example_dataset(input_dir)

    st.write(
        """
    #####
    Before you proceed with the next tasks you need to create a .csv table.
    The table should contain the following information for each case:
    - **ID** (has to be unique)
    - Path to the **image**
    - Path to the **segmentation**
    - Optionally: **Label** (0 or 1)

    Example:
    """
    )
    example_path_df = pd.read_csv(
        Path(__file__).parents[1] / "paths_example.csv"
    )
    st.write(example_path_df)
    st.write(
        """
    #####
    You can create the table manually (e.g. in Excel) or, if your dataset
    follows the specified structure, you can **generate it below** ⬇
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
    formats = list(dir_structures.keys())
    col1, col2 = st.columns(2)
    with col1:
        format = st.radio("Dataset structure:", formats)
    with col2:
        st.write(dir_structures[format])
    data_dir = st.text_input(
        "Enter path to the root directory of your dataset:",
        value=input_dir,
    )
    if not data_dir:
        st.stop()
    data_dir = Path(data_dir).absolute()
    validation_utils.check_if_dir_exists(data_dir)
    col1, col2 = st.columns(2)

    if format == formats[0] or format == formats[1]:
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
    elif format == formats[1]:
        paths_df = preprocessing.get_paths_with_separate_folder_per_case_loose(
            data_dir=data_dir,
            image_stem=image_stem,
            mask_stem=mask_stem,
        )
    else:
        with col1:
            image_dir = st.text_input(
                "Enter path to the folder with images:",
                value=data_dir / "images",
            )
        with col2:
            segmentation_dir = st.text_input(
                "Enter path to the folder with segmentations:",
                value=data_dir / "segmentations",
            )
        paths_df = preprocessing.get_paths_with_separate_image_seg_folders(
            image_dir=image_dir,
            mask_dir=segmentation_dir,
        )
    if paths_df.empty:
        st.warning(
            "Looks like no files were found."
            " Please check the dataset root directory and file names."
        )
        st.stop()
    else:
        st.success(f"Found {len(paths_df)} cases.")
        st.dataframe(paths_df)
        st_read.save_table_streamlit(paths_df, Path(result_dir) / "paths.csv")

    st_utils.next_step("1.2_Prepare_dataset")


if __name__ == "__main__":
    show()
