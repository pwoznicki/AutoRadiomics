from pathlib import Path

import streamlit as st

from autorad.feature_extraction import FeatureExtractor
from autorad.utils import io
from autorad.webapp import extraction_utils, st_read, st_utils


def show():
    st_utils.show_title()
    st.header("Feature extraction")
    st.info(
        """
        In this step, you can extract quantitative imaging features from your dataset.
        For that you'll need the table with paths to every image and segmentation, generated in the previous step.

        Expected input:
            CSV table with with absolute paths to the image and the mask
            for each case.
    """
    )
    result_dir = Path(st_read.get_result_dir())
    dataset = st_read.load_path_df(input_dir=result_dir)

    with st.expander("Inspect the data"):
        if st.button("Show random case"):
            st_read.show_random_case(dataset)

    extraction_params = extraction_utils.radiomics_params()
    col1, col2 = st.columns(2)
    with col1:
        filename = st.text_input(
            "Name for the table with features", "features.csv"
        )
    with col2:
        n_jobs = st.slider(
            "Number of threads", min_value=1, max_value=8, value=1
        )
    extraction_button = st.button("Run feature extraction")
    if extraction_button:
        params_path = result_dir / "extraction_params.json"
        io.save_json(extraction_params, params_path)
        extractor = FeatureExtractor(
            dataset=dataset,
            n_jobs=n_jobs,
            extraction_params=params_path,
        )
        with st.spinner("Extracting features"):
            feature_df = extractor.run()
        st.dataframe(feature_df)
        st_read.save_table_streamlit(
            feature_df,
            result_dir / filename,
            button=False,
        )

    st_utils.next_step("2.1_Train_models")


if __name__ == "__main__":
    show()
