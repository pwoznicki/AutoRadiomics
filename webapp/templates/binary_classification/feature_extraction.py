from pathlib import Path
import streamlit as st

from extractor import StreamlitFeatureExtractor
import utils
from template_utils import radiomics_params
from classrad.config import config
from classrad.data.dataset import ImageDataset


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
    st.write(path_df)
    # path_df.replace("", np.nan, inplace=True)
    col1, col2, col3 = st.columns(3)
    colnames = path_df.columns.tolist()
    with col1:
        image_col = st.selectbox("Path to image", colnames)
    with col2:
        mask_col = st.selectbox("Path to segmentation", colnames)
    with col3:
        id_col = st.selectbox("ID column", colnames)
    path_df.dropna(subset=[image_col, mask_col], inplace=True)
    image_dataset = ImageDataset().from_dataframe(
        df=path_df,
        image_colname=image_col,
        mask_colname=mask_col,
        id_colname=id_col,
    )
    radiomics_params()

    col1, col2 = st.columns(2)
    with col1:
        num_threads = st.slider(
            "Number of threads", min_value=1, max_value=8, value=1
        )
    with col2:
        out_fname = st.text_input(
            "Name for the resulting table", "features_test"
        )
    out_path = Path(config.RESULT_DIR) / f"{out_fname}.csv"

    start_extraction = st.button("Run feature extraction")

    if start_extraction:
        progressbar = st.progress(0)
        extractor = StreamlitFeatureExtractor(
            dataset=image_dataset,
            out_path=out_path,
            verbose=True,
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
