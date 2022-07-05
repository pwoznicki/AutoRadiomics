import base64
import logging
import math
import os
import re
import uuid
from pathlib import Path

import jupytext
import radiomics
import SimpleITK as sitk
import streamlit as st
from radiomics import featureextractor

from autorad.config import config
from autorad.data.dataset import ImageDataset
from autorad.webapp import utils


def guess_idx_of_img_colname(colnames):
    for i, colname in enumerate(colnames):
        if "img" in colname or "image" in colname:
            return i
    return 0


def guess_idx_of_seg_colname(colnames):
    for i, colname in enumerate(colnames):
        if "seg" in colname or "mask" in colname:
            return i
    return 0


def load_path_df():
    path_df = utils.load_df("Choose a CSV file with paths:")
    st.dataframe(path_df)
    col1, col2, col3 = st.columns(3)
    colnames = path_df.columns.tolist()
    with col1:
        image_col = st.selectbox(
            "Path to image", colnames, index=guess_idx_of_img_colname(colnames)
        )
    with col2:
        mask_col = st.selectbox(
            "Path to segmentation",
            colnames,
            index=guess_idx_of_seg_colname(colnames),
        )
    with col3:
        id_col = st.selectbox("ID (optional)", [None] + colnames)
    path_df.dropna(subset=[image_col, mask_col], inplace=True)
    dataset = ImageDataset(
        df=path_df,
        image_colname=image_col,
        mask_colname=mask_col,
        ID_colname=id_col,
        root_dir=config.INPUT_DIR,
    )
    return dataset


def extract_feature_maps(image_path, seg_path, save_dir):
    extraction_params = radiomics_params_voxelbased()
    radiomics.setVerbosity(logging.INFO)
    extractor = featureextractor.RadiomicsFeatureExtractor(extraction_params)
    feature_vector = extractor.execute(image_path, seg_path, voxelBased=True)
    for feature_name, feature_value in feature_vector.items():
        if isinstance(feature_value, sitk.Image):
            save_path = save_dir / f"{feature_name}.nii.gz"
            save_path = str(save_path)
            sitk.WriteImage(feature_value, save_path)


def radiomics_params():
    param_dir = Path(config.PARAM_DIR)
    presets = config.PRESETS
    preset_options = list(presets.keys())
    name = st.selectbox(
        "Choose a preset with parameters for feature extraction",
        preset_options,
    )
    preset_setup = utils.read_yaml(param_dir / presets[name])
    final_setup = preset_setup.copy()

    with st.expander("Manually edit the extraction parameters"):
        final_setup = select_classes(preset_setup, final_setup)
        setting = preset_setup["setting"]
        st.write(""" Filters: """)
        filter = preset_setup["imageType"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("original", value=("Original" in filter))
        with col2:
            turn_on_log = st.checkbox(
                "Laplacian of Gaussian", value=("LoG" in filter)
            )
            if turn_on_log:
                sigmas = filter["LoG"]["sigma"]
                for sigma in sigmas:
                    st.number_input("sigma", value=sigma)
        with col3:
            st.checkbox("Wavelet", value=("Wavelet" in filter))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(""" ##### Normalization: """)
            normalization = setting["normalize"]
            is_normalized = st.checkbox("Normalize", value=normalization)
            if is_normalized:
                final_setup["setting"]["normalize"] = True
            else:
                final_setup["setting"]["normalize"] = False
        with col2:
            bin_width = st.number_input(
                """Bin width:""", value=setting["binWidth"]
            )
            final_setup["setting"]["binWidth"] = bin_width
        with col3:
            label = st.number_input("Label:", value=setting["label"])
            final_setup["setting"]["label"] = int(label)
        st.write(""" #### Full parameter file: """, preset_setup)
    return preset_setup


def choose_preset():
    config_dir = os.path.dirname(config.__file__)
    param_dir = Path(config_dir) / "pyradiomics_params"
    presets = config.PRESETS
    preset_options = list(presets.keys())
    name = st.selectbox(
        "Choose a preset with parameters for feature extraction",
        preset_options,
    )
    preset_setup = utils.read_yaml(param_dir / presets[name])
    return preset_setup


def update_setup_with_class(setup, class_name, class_active):
    assert "featureClass" in setup
    if class_active[class_name]:
        setup["featureClass"][class_name] = []
    else:
        setup["featureClass"].pop(class_name, None)
    return setup


def select_classes(preset_setup, final_setup, exclude_shape=False):
    st.write(""" #### Select classes: """)

    all_feature_names = config.PYRADIOMICS_FEATURE_NAMES
    all_classes = list(all_feature_names.keys())
    if exclude_shape:
        if "shape" in all_classes:
            all_classes.remove("shape")
    preset_classes = preset_setup["featureClass"]
    class_active = {}
    cols = st.columns(len(all_classes))
    for i, class_name in enumerate(all_classes):
        with cols[i]:
            class_active[class_name] = st.checkbox(
                class_name, value=(class_name in preset_classes), key=i
            )
        final_setup = update_setup_with_class(
            final_setup, class_name, class_active
        )
    # st.write("##### Optional: select specific features within class:")
    st.info(
        """
        If you select a class, all its features are included by default.
        You can however select only specific features (optional):
        """
    )
    cols = st.columns(len(preset_classes.keys()))
    for i, class_name in enumerate(preset_classes.keys()):
        with cols[i]:
            if class_active[class_name]:
                feature_names = st.multiselect(
                    class_name, all_feature_names[class_name], key=class_name
                )
                final_setup["featureClass"][class_name] = feature_names
    return final_setup


def radiomics_params_voxelbased():
    param_dir = Path(config.PARAM_DIR)
    presets = config.PRESETS
    preset_options = list(presets.keys())
    name = st.selectbox(
        "Choose a preset with parameters for feature extraction",
        preset_options,
    )
    preset_setup = utils.read_yaml(param_dir / presets[name])
    if "shape" in preset_setup["featureClass"]:
        preset_setup["featureClass"].pop("shape", None)
    final_setup = preset_setup.copy()

    with st.expander("Manually edit the extraction parameters"):
        final_setup = select_classes(
            preset_setup, final_setup, exclude_shape=True
        )

        setting = preset_setup["setting"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(""" ##### Normalization: """)
            normalization = setting["normalize"]
            is_normalized = st.checkbox("Normalize", value=normalization)
            if is_normalized:
                final_setup["setting"]["normalize"] = True
            else:
                final_setup["setting"]["normalize"] = False
        with col2:
            bin_width = st.number_input(
                """Bin width:""", value=setting["binWidth"]
            )
            final_setup["setting"]["binWidth"] = bin_width
        with col3:
            label = st.number_input("Label:", value=setting["label"])
            final_setup["setting"]["label"] = int(label)
        st.write(""" #### Full parameter file: """, preset_setup)

    return final_setup


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


# borrowed from
# https://github.com/jrieke/traingenerator/blob/main/app/utils.py
def download_button(
    object_to_download, download_filename, button_text  # , pickle_it=False
):
    """
    Generates a link to download the given object_to_download.
    """
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )

    st.markdown(dl_link, unsafe_allow_html=True)
