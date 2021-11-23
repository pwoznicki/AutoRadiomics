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
        turn_on_original = st.checkbox("original", value=("Original" in filter))
    with col2:
        turn_on_log = st.checkbox("Laplacian of Gaussian", value=("LoG" in filter))
        if turn_on_log:
            sigmas = filter["LoG"]["sigma"]
            for sigma in sigmas:
                st.number_input("sigma", value=sigma)
    with col3:
        turn_on_wavelet = st.checkbox("Wavelet", value=("Wavelet" in filter))
    classes = setup["featureClass"]
    all_classes = ["firstorder", "shape", "glcm", "glszm", "glrlm", "gldm"]
    st.write(""" Classes: """)
    cols = st.columns(6)
    for i in range(len(all_classes)):
        with cols[i]:
            st.checkbox(all_classes[i], value=True, key=i)
    setting = setup["setting"]
    st.write(""" Normalization: """)
    normalization = setting["normalize"]
    st.checkbox("Normalize", value=normalization)
    st.number_input("Bin width", value=setting["binWidth"])
    st.write(""" Full parameter file: """, setup)


def radiomics_params_voxelbased():
    config_dir = os.path.dirname(config.__file__)
    param_dir = Path(config_dir) / "pyradiomics_params"
    presets = {"default CT": "default_feature_map.yaml"}
    preset_options = list(presets.keys())
    name = st.selectbox(
        "Choose a preset with parameters for feature extraction", preset_options
    )
    preset_setup = utils.read_yaml(param_dir / presets[name])
    final_setup = preset_setup.copy()
    with st.expander("Manually edit the extraction parameters"):
        classes = preset_setup["featureClass"]
        all_classes = ["firstorder", "glcm", "glszm", "glrlm", "gldm", "ngtdm"]
        st.write(""" Classes: """)
        cols = st.columns(len(all_classes))
        feature_active = {}
        for i, feature_class in enumerate(all_classes):
            with cols[i]:
                feature_active[feature_class] = st.checkbox(
                    feature_class, value=(feature_class in classes), key=i
                )
            if feature_active[feature_class]:
                final_setup["featureClass"][feature_class] = []
            elif feature_class in final_setup["featureClass"]:
                final_setup["featureClass"].pop(feature_class)
        setting = preset_setup["setting"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(""" Normalization: """)
            normalization = setting["normalize"]
            is_normalized = st.checkbox("Normalize", value=normalization)
            if is_normalized:
                final_setup["setting"]["normalize"] = True
            else:
                final_setup["setting"]["normalize"] = False
        with col2:
            bin_width = st.number_input("Bin width:", value=setting["binWidth"])
            final_setup["setting"]["binWidth"] = bin_width
        with col3:
            label = st.number_input("Label:", value=setting["label"])
            final_setup["setting"]["label"] = int(label)
        st.write(""" Full parameter file: """, preset_setup)

    return final_setup
