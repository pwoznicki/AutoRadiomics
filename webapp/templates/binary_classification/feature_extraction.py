from pathlib import Path

import seaborn as sns
import streamlit as st

from classrad.config import config
from classrad.data.dataset import ImageDataset
from webapp import utils
from webapp.extractor import StreamlitFeatureExtractor
from webapp.template_utils import radiomics_params


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
    col1, col2, col3 = st.columns(3)
    colnames = path_df.columns.tolist()
    with col1:
        image_col = st.selectbox("Path to image", colnames)
    with col2:
        mask_col = st.selectbox("Path to segmentation", colnames)
    with col3:
        id_col = st.selectbox("ID (optional)", [None] + colnames)
    path_df.dropna(subset=[image_col, mask_col], inplace=True)
    radiomics_params()
    result_dir = Path(config.RESULT_DIR)
    result_dir.mkdir(exist_ok=True)
    out_path = result_dir / "features.csv"
    num_threads = st.slider(
        "Number of threads", min_value=1, max_value=8, value=1
    )
    start_extraction = st.button("Run feature extraction")
    if start_extraction:
        progressbar = st.progress(0)
        dataset = ImageDataset(
            df=path_df,
            image_colname=image_col,
            mask_colname=mask_col,
            ID_colname=id_col,
            root_dir=config.INPUT_DIR,
        )
        extractor = StreamlitFeatureExtractor(
            dataset=dataset,
            out_path=out_path,
            verbose=False,
        )
        feature_df = extractor.extract_features(
            num_threads=num_threads, progressbar=progressbar
        )
        st.success(
            f"Done! Features saved in your result directory ({out_path})"
        )
        feature_colnames = [
            col
            for col in feature_df.columns
            if col.startswith(("original", "wavelet", "shape"))
        ]
        cm = sns.light_palette("green", as_cmap=True)
        display_df = (
            feature_df.copy()
            .loc[:, feature_colnames]
            .astype(float)
            .style.background_gradient(cmap=cm)
        )
        st.dataframe(display_df)


if __name__ == "__main__":
    show()
