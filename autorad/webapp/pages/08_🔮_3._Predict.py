from datetime import datetime
from io import BytesIO
from pathlib import Path

import shap
import streamlit as st

from autorad.inference import infer, infer_utils
from autorad.visualization import plot_volumes
from autorad.webapp import st_read, st_utils


def show():
    st_utils.show_title()
    input_dir = st_read.get_input_dir()
    result_dir = st_read.get_result_dir()

    artifacts = st_utils.select_model()

    st_read.find_all_data(input_dir)
    img_path, mask_path = st_read.read_image_seg_paths()
    run_prediction = st.button("Predict")
    if not run_prediction:
        st.stop()
    if not img_path or not mask_path:
        st.error("Image or segmentation not found.")

    inferrer = infer.Inferrer(
        extraction_config=artifacts["extraction_config"],
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"],
        result_dir=result_dir,
    )
    with st.spinner("Predicting"):
        (
            feature_df,
            X_preprocessed,
            result,
        ) = inferrer.predict_proba_with_features(img_path, mask_path)
    st.write(f"For provided case probability of class = 1 is: {result:.2f}")

    tab1, tab2, tab3 = st.tabs(["Explanation", "Image", "Radiomics features"])
    with tab1:
        shap_values = artifacts["explainer"](
            X_preprocessed,  # max_evals=2 * X_preprocessed.shape[1] + 1
        )
        shap.initjs()
        shap_fig = shap.plots.waterfall(
            shap_values[0], max_display=10, show=False
        )
        st.set_option("deprecation.showPyplotGlobalUse", False)
        try:
            # st.pyplot(shap_fig)
            buf = BytesIO()
            shap_fig.savefig(buf, format="png", bbox_inches="tight")
            st.image(buf)
        except ValueError as e:
            st.error("Failed to plot SHAP explanation.")
            st.error(e)
    with tab2:
        try:
            img_fig = plot_volumes.plot_roi(img_path, mask_path)
            img_fig.update_layout(width=400, height=400)
            st.plotly_chart(img_fig)
        except TypeError:
            st.warning("Failed to plot image and segmentation.")
    with tab3:
        st.write(feature_df)
        st.download_button(
            "Download features ⬇️",
            feature_df.to_csv(),
            f"features_{datetime.now()}.csv",
        )


if __name__ == "__main__":
    show()
