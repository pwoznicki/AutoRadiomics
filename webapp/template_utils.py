import os
from pathlib import Path

import streamlit as st
import utils

from classrad.config import config


def radiomics_params():
    config_dir = os.path.dirname(config.__file__)
    param_dir = Path(config_dir) / "pyradiomics_params"
    presets = {"CT from Baessler et al. (2019)": "Baessler_CT.yaml"}
    preset_options = list(presets.keys())
    name = st.selectbox("Choose a preset", preset_options)
    setup = utils.read_yaml(param_dir / presets[name])
    st.write(""" Filters: """)
    filter = setup["imageType"]
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
    all_classes = ["firstorder", "shape", "glcm", "glszm", "glrlm", "gldm"]
    st.write(""" Classes: """)
    cols = st.columns(len(all_classes))
    for i in range(len(all_classes)):
        with cols[i]:
            st.checkbox(all_classes[i], value=True)
    # with cols[5]:
    #    st.checkbox("glcm", value=True)
    setting = setup["setting"]
    st.write(""" Normalization: """)
    normalization = setting["normalize"]
    st.checkbox("Normalize", value=normalization)
    st.number_input("Bin width", value=setting["binWidth"])
    st.write(""" Full parameter file: """, setup)


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


def select_classes(preset_setup, final_setup):
    st.write(""" #### Select classes: """)

    all_feature_names = config.PYRADIOMICS_FEATURE_NAMES
    all_classes = all_feature_names.keys()
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
        final_setup = select_classes(preset_setup, final_setup)

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
