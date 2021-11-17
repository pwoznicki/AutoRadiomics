import streamlit as st
from pathlib import Path
from extractor import StreamlitFeatureExtractor
import utils

result_dir = Path("/Users/p.woznicki/Documents/test")


def radiomics_params():
    param_dir = Path("../Radiomics/feature_extraction/default_params")
    presets = {"CT from Baessler et al. (2019)": "pyradiomics.yaml"}
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
    st.write(""" Full parameter file: """, setup)


def show():
    with st.sidebar:
        st.write(
            """
            Expected input:  
                CSV file with with absolute paths to the image and the mask for each case.
        """
        )
    path_df = utils.load_df("Choose a CSV file with paths:")
    st.write(path_df)
    # path_df.replace("", np.nan, inplace=True)
    col1, col2 = st.columns(2)
    colnames = path_df.columns.tolist()
    with col1:
        image_col = st.selectbox("Path to image", colnames)
    with col2:
        mask_col = st.selectbox("Path to segmentation", colnames)
    path_df.dropna(subset=[image_col, mask_col], inplace=True)
    radiomics_params()

    out_path = result_dir / "features_test.csv"
    num_threads = st.slider("Number of threads", min_value=1, max_value=8, value=1)
    start_extraction = st.button("Run feature extraction")

    if start_extraction:
        progressbar = st.progress(0)
        extractor = StreamlitFeatureExtractor(
            df=path_df.head(2),
            out_path=out_path,
            image_col=image_col,
            mask_col=mask_col,
            verbose=False,
            num_threads=num_threads,
        )
        feature_df = extractor.extract_features(progressbar=progressbar)
        st.success(f"Done! Features saved in your result directory ({out_path})")
        feature_colnames = [
            col
            for col in feature_df.columns
            if col.startswith(("original", "wavelet", "shape"))
        ]
        feature_df[feature_colnames] = feature_df[feature_colnames].astype(float)
        st.write(feature_df)


if __name__ == "__main__":
    show()
