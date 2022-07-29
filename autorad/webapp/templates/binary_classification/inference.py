import os

import streamlit as st
import template_utils

from autorad.config import config


def file_selector(folder_path="."):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)


def show():
    # model = mlflow.pyfunc.load_model(model_path)

    filename = file_selector()
    st.write("You selected `%s`" % filename)
    col1, col2 = st.columns(2)
    with col1:
        img_path = st.file_uploader(label="Input image")
    with col2:
        mask_path = st.file_uploader(label="Input segmentation")
    # model.predict()
    save_dir = config.RESULT_DIR
    if st.button("Predict"):
        st.write("Predicting...")
        template_utils.extract_feature_maps(img_path, mask_path, save_dir)


if __name__ == "__main__":
    show()
