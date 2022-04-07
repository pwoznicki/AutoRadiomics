from pathlib import Path

import streamlit as st
import utils
from extractor import StreamlitFeatureExtractor
from template_utils import radiomics_params

from classrad.config import config


def show():
    with st.sidebar:
        st.write(
            """
            Expected input:
                CSV file with with absolute paths to the image and the mask
                for each case.
        """
        )
    path_df = utils.load_df("Choose a CSV file with paths:")
    st.dataframe(path_df)
    # path_df.replace("", np.nan, inplace=True)
    col1, col2 = st.columns(2)
    colnames = path_df.columns.tolist()
    with col1:
        image_col = st.selectbox("Path to image", colnames)
    with col2:
        mask_col = st.selectbox("Path to segmentation", colnames)
    path_df.dropna(subset=[image_col, mask_col], inplace=True)
    radiomics_params()

    out_path = Path(config.RESULT_DIR) / "features_test.csv"
    num_threads = st.slider(
        "Number of threads", min_value=1, max_value=8, value=1
    )
    start_extraction = st.button("Run feature extraction")

    if start_extraction:
        progressbar = st.progress(0)
        extractor = StreamlitFeatureExtractor(
            df=path_df.head(2),
            out_path=out_path,
            image_col=image_col,
            mask_col=mask_col,
            #  extraction_params=extraction_params,
            verbose=False,
            num_threads=num_threads,
        )
        feature_df = extractor.extract_features(progressbar=progressbar)
        st.success(
            f"Done! Features saved in your result directory ({out_path})"
        )
        feature_colnames = [
            col
            for col in feature_df.columns
            if col.startswith(("original", "wavelet", "shape"))
        ]
        feature_df[feature_colnames] = feature_df[feature_colnames].astype(
            float
        )
        st.write(feature_df)


if __name__ == "__main__":
    show()
