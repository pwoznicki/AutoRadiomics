import itertools
import math
import os
from pathlib import Path

import jupytext
import pandas as pd
import streamlit as st

from autorad.config.type_definitions import PathLike
from autorad.data import ImageDataset
from autorad.visualization import plot_volumes


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
    if not filenames:
        st.error(f"No tables found in this folder: {str(dir_path)}")
        st.stop()
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


def save_table(df, save_path):
    df.to_csv(save_path, index=False)
    st.success(f"Done! Table saved ({save_path})")


def save_table_streamlit(df, save_path, button=True):
    if button:
        saved = st.button("Looks good? Save the table ⬇️")
    else:
        saved = True
    if saved:
        df.to_csv(save_path, index=False)
        st.success(f"Done! Table saved in your result directory ({save_path})")


def get_input_dir():
    if "AUTORAD_INPUT_DIR" in os.environ:
        return os.environ["AUTORAD_INPUT_DIR"]
    else:
        st.error(
            "Oops, input directory not set! Go to the config page and set it."
        )
        st.stop()


def get_result_dir():
    if "AUTORAD_RESULT_DIR" in os.environ:
        return os.environ["AUTORAD_RESULT_DIR"]
    else:
        st.error(
            "Oops, result directory not set! Go to the config page and set it."
        )
        st.stop()


def dir_nonempty(dir_path):
    path = Path(dir_path)
    if not path.is_dir():
        return False
    has_next = next(path.iterdir(), None)
    if has_next is None:
        return False
    return True
