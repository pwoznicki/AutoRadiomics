from pathlib import Path
import pandas as pd
import streamlit as st


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
    example_path_df = pd.read_csv("paths_example.csv")
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
    col1, col2 = st.columns(2)
    with col1:
        format = st.selectbox("Dataset structure", dir_structures.keys())
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
        st.success(f"The entered path is a directory. ({data_dir})")


if __name__ == "__main__":
    show()
