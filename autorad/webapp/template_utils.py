import itertools
import math
import os
import shutil
from pathlib import Path

import jupytext
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from autorad.config.type_definitions import PathLike
from autorad.data.dataset import ImageDataset
from autorad.utils import conversion, spatial
from autorad.visualization import plot_volumes
from autorad.webapp import utils


def show_title():
    st.set_page_config(
        page_title="AutoRadiomics",
        layout="wide",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.title("AutoRadiomics")
    with col2:
        st.write(
            """
        ####
        The easiest framework for experimenting
        using `pyradiomics` and `scikit-learn`.
        """
        )


def next_step(title):
    _, _, col = st.columns(3)
    with col:
        next_page = st.button(
            f"Go to the next step ➡️ {title.replace('_', ' ')}"
        )
        if next_page:
            switch_page(title)


def find_all_data(input_dir):
    input_dir = Path(input_dir)
    files = itertools.chain(
        input_dir.rglob("*.nii*"), input_dir.rglob("*.nrrd")
    )
    files = [str(f) for f in files]
    st.write("""Files found in your input directory:""")
    st.write(files)


def read_image_seg_paths():
    col1, col2 = st.columns(2)
    with col1:
        image_path = st.text_input("Path to the image:")
        image_path = image_path.strip('"')
        if os.path.isfile(image_path):
            st.success("Image found!")
    with col2:
        seg_path = st.text_input("Path to the segmentation:")
        seg_path = seg_path.strip('"')
        if os.path.isfile(seg_path):
            st.success("Segmentation found!")

    return image_path, seg_path


def copy_images_to_nnunet(
    input_dir: PathLike,
    output_dir: PathLike,
    in_filename: str = "image.nii.gz",
):
    files = Path(input_dir).rglob(f"*{in_filename}")

    for file in files:
        output_path = Path(output_dir) / f"{file.parent.name}_0000.nii.gz"
        shutil.copy(str(file), str(output_path))


def copy_predictions_from_nnunet(
    pred_dir: PathLike,
    out_dir: PathLike,
    out_filename: str = "segmentation_auto.nii.gz",
):
    files = Path(pred_dir).rglob("*.nii.gz")

    for file in files:
        id_ = file.stem.split("_")[0]
        output_path = Path(out_dir) / id_ / out_filename
        shutil.copy(str(file), str(output_path))


def dicom_to_nifti_expander(data_dir):
    st.write(
        """
    For the later analysis, the images (and the segmentations) need to be
    in the **NIfTI format (.nii.gz)**. \\
    If you have DICOMs, convert them here:
    """
    )
    with st.expander("Convert DICOM to NIFTI"):
        dicom_dir = st.text_input(
            "Choose the root directory with DICOMs:",
            data_dir,
        )
        dicom_dir = Path(dicom_dir)
        if dicom_dir.is_dir():
            subdirs = [d.stem for d in dicom_dir.iterdir() if d.is_dir()]
            st.success(
                f"Directory exists and contains {len(subdirs)} subfolders. Check below if they look good.",
            )
            st.json(subdirs)
        else:
            st.error(
                f"Directory {dicom_dir} does not exist",
            )
        out_dirname = st.text_input("Output directory name:", "nifti")
        out_dir = Path(data_dir) / out_dirname
        out_dir.mkdir(exist_ok=True)
        if len(os.listdir(out_dir)) > 0:
            st.warning(f"Output directory {out_dir} is not empty. ")
        multiple_images = st.checkbox(
            "I have multiple DICOM images per patient (will require manual renaming)",
            value=False,
        )
        if not multiple_images:
            out_filename = "image"
        else:
            out_filename = dicom_dir.name + ".nii.gz"
        run_conversion = st.button("Convert")
        if run_conversion:
            with st.spinner("Converting DICOMs to NIFTI..."):
                conversion.convert_to_nifti(
                    dicom_dir, output_path=out_dir / out_filename
                )
            st.success(f"Done! Nifties saved in {out_dir}")

        return out_dir


def leave_only_organ_segmentation(
    seg_dir: PathLike,
    organ_label: int,
    save_dir: PathLike,
):
    """Takes the segmentations from a trained nnUNet model,
    filters out only the segmentation for `organ_label`, to which label=1 is assigned.
    The other segmentations are cleared (label=0).
    """
    label_map = {organ_label: 1}
    for mask_path in Path(seg_dir).glob("*.nii.gz"):
        save_path = Path(save_dir) / mask_path.name
        spatial.relabel_mask(
            mask_path=mask_path,
            label_map=label_map,
            save_path=save_path,
            strict=False,
            set_rest_to_zero=True,
        )


def file_selector(
    dir_path: PathLike, text: str, ext: str | list[str] | None = None
):
    filenames = os.listdir(dir_path)
    if ext is None:
        filenames = [
            f for f in filenames if os.path.isfile(os.path.join(dir_path, f))
        ]
    else:
        if isinstance(ext, str):
            ext = [ext]
        filenames = [f for f in filenames if f.split(".")[-1] in ext]
    selected_filename = st.selectbox(text, filenames)
    return os.path.join(dir_path, selected_filename)


guess_dict = {
    "image": ["img", "image"],
    "segmentation": ["seg", "mask"],
    "id": ["id"],
    "label": ["label", "diagnos"],
}


def guess_idx_of_column(columns, coltype):
    for i, colname in enumerate(columns):
        if any([p in colname.lower() for p in guess_dict[coltype]]):
            return i
    return 0


def load_df(data_dir, label):
    uploaded_file = file_selector(
        data_dir,
        label,
        ext=["csv", "xlsx", "xls"],
    )
    if uploaded_file.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.endswith(".xlsx") or uploaded_file.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unknown file type")
    return df


def load_path_df(input_dir):
    # input_dir = utils.get_input_dir()
    # result_dir = utils.get_result_dir()
    path_df = load_df(input_dir, "Choose a CSV table with paths:")
    st.dataframe(path_df)
    col1, col2, col3 = st.columns(3)
    colnames = path_df.columns.tolist()
    with col1:
        image_col = st.selectbox(
            "Path to image",
            colnames,
            index=guess_idx_of_column(colnames, "image"),
        )
    with col2:
        mask_col = st.selectbox(
            "Path to segmentation",
            colnames,
            index=guess_idx_of_column(colnames, "segmentation"),
        )
    with col3:
        ID_options = ["None"] + colnames
        id_col = st.selectbox(
            "ID (optional)",
            ID_options,
            index=guess_idx_of_column(ID_options, "id"),
        )
    path_df.dropna(subset=[image_col, mask_col], inplace=True)
    dataset = ImageDataset(
        df=path_df,
        image_colname=image_col,
        mask_colname=mask_col,
        ID_colname=id_col,
        root_dir=input_dir,
    )
    return dataset


def show_random_case(dataset: ImageDataset):
    try:
        row = dataset.df.sample(1).iloc[0]
    except IndexError:
        raise IndexError("No cases found")
    st.dataframe(row)
    image_path = row[dataset.image_colname]
    mask_path = row[dataset.mask_colname]
    try:
        fig = plot_volumes.plot_roi(image_path, mask_path)
        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)
    except TypeError:
        raise TypeError(
            "Image or mask path is not a string. "
            "Did you correctly set the paths above?"
        )


def code_header(text):
    """
    Insert section header into a jinja file, formatted as Python comment.
    Leave 2 blank lines before the header.
    """
    seperator_len = (75 - len(text)) / 2
    seperator_len_left = math.floor(seperator_len)
    seperator_len_right = math.ceil(seperator_len)
    return f"# {'-' * seperator_len_left} {text} {'-' * seperator_len_right}"


def notebook_header(text):
    """
    Insert section header into a jinja file, formatted as notebook cell.
    Leave 2 blank lines before the header.
    """
    return f"""# # {text}
"""


def to_notebook(code):
    """Converts Python code to Jupyter notebook format."""
    notebook = jupytext.reads(code, fmt="py")
    return jupytext.writes(notebook, fmt="ipynb")
