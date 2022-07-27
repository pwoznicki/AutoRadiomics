from pathlib import Path

import seaborn as sns
import streamlit as st

from autorad.config import config
from autorad.utils import io
from autorad.visualization import plot_volumes
from autorad.webapp import template_utils
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
    result_dir = Path(config.RESULT_DIR)
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
    out_path = result_dir / "features.csv"
    n_jobs = st.slider("Number of threads", min_value=1, max_value=8, value=1)
    start_extraction = st.button("Run feature extraction")
    if start_extraction:
        params_path = result_dir / "extraction_params.json"
        io.save_json(extraction_params, params_path)
        extractor = StreamlitFeatureExtractor(
            dataset=dataset,
            n_jobs=n_jobs,
            extraction_params=params_path,
        )
        with st.spinner("Extracting features"):
            feature_df = extractor.run()
        feature_df.to_csv(out_path, index=False)
        st.success(
            f"Done! Features saved in your result directory ({out_path})"
        )
        cm = sns.light_palette("green", as_cmap=True)
        display_df = feature_df.style.background_gradient(cmap=cm)
        st.dataframe(display_df)
        st.download_button(
            label="Download features ⬇️",
            data=feature_df.to_csv(index=False),
            file_name="features.csv",
        )


if __name__ == "__main__":
    show()
