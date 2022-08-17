import streamlit as st

from autorad.webapp import template_utils, utils


def show():
    # model = mlflow.pyfunc.load_model(model_path)
    input_dir = utils.get_input_dir()
    filename = template_utils.file_selector(input_dir, "Select image")
    st.write("You selected `%s`" % filename)
    col1, col2 = st.columns(2)
    with col1:
        img_path = st.file_uploader(label="Input image")
    with col2:
        mask_path = st.file_uploader(label="Input segmentation")
    # model.predict()
    save_dir = utils.get_result_dir()
    if st.button("Predict"):
        st.write("Predicting...")
        template_utils.extract_feature_maps(img_path, mask_path, save_dir)


if __name__ == "__main__":
    show()
