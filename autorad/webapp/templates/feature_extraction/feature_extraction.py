from pathlib import Path

import streamlit as st

from autorad.utils import io
from autorad.visualization import plot_volumes
from autorad.webapp import template_utils, utils
from autorad.webapp.extractor import StreamlitFeatureExtractor
from autorad.webapp.template_utils import radiomics_params


def show():
    st.header("Feature extraction")
    with st.sidebar:
        st.write(
            """
            Expected input:
                CSV file with with absolute paths to the image and the mask
                for each case.
        """
        )
    dataset = template_utils.load_path_df()
    result_dir = Path(utils.get_result_dir())
    result_dir.mkdir(exist_ok=True)
    with st.expander("Inspect the data"):
        if st.button("Draw random case"):
            row = dataset.df.sample(1)
            st.dataframe(row)
            image_path = row[dataset.image_colname]
            mask_path = row[dataset.mask_colname]
            try:
                fig = plot_volumes.plot_roi(image_path, mask_path)
                fig.update_layout(width=300, height=300)
                st.plotly_chart(fig)
            except TypeError:
                raise ValueError(
                    "Image or mask path is not a string. "
                    "Did you correctly set the paths above?"
                )

    extraction_params = radiomics_params()
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
        extractor = StreamlitFeatureExtractor(
            dataset=dataset,
            n_jobs=n_jobs,
            extraction_params=params_path,
        )
        with st.spinner("Extracting features"):
            feature_df = extractor.run()
        st.dataframe(feature_df)
        utils.save_table_in_result_dir(feature_df, filename, button=False)


if __name__ == "__main__":
    show()
