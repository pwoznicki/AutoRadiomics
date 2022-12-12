from datetime import datetime
from io import BytesIO
from pathlib import Path

import shap
import streamlit as st

from autorad.inference import infer, infer_utils
from autorad.utils import mlflow_utils
from autorad.visualization import plot_volumes
from autorad.webapp import template_utils, utils


def show():
    template_utils.show_title()
    input_dir = utils.get_input_dir()
    result_dir = utils.get_result_dir()
    st.subheader("Select model")
    start_mlflow = st.button("Browse trained models in MLFlow")
    if start_mlflow:
        mlflow_utils.start_mlflow_server()
        mlflow_utils.open_mlflow_dashboard()

    selection_modes = ["Select best model"]
    st.radio(
        "Do you want to use the best model trained, or select one yourself?",
        selection_modes,
    )
    # TODO: add option to select model yourself
    # if mode_choice == selection_modes[1]:
    #     st.text_input(
    #         "Paste here path to the selected run",
    #         help="Click on `Browse trained models`, open selected run and copy its `Full path`",
    #    )
    template_utils.find_all_data(input_dir)
    img_path, mask_path = template_utils.read_image_seg_paths()
    run_prediction = st.button("Predict")
    if not run_prediction:
        st.stop()
    if not img_path or not mask_path:
        st.error("Image or segmentation not found.")

    artifacts = infer_utils.get_artifacts_from_best_run(
        experiment_name="model_training"
    )

    inferrer = infer.Inferrer(
        extraction_config=artifacts["extraction_config"],
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"],
        result_dir=result_dir,
    )
    with st.spinner("Extracting radiomics features..."):
        feature_df = infer.infer_radiomics_features(
            img_path,
            mask_path,
            artifacts["extraction_config"],
        )
    feature_df.to_csv(Path(result_dir) / "infer_df.csv", index=False)
    with st.spinner("Predicting..."):
        result = inferrer.predict_proba(feature_df)
    st.write(f"For provided case probability of class=1 is: {result[0]:.2f}")
    shap.initjs()
    X_preprocessed = inferrer.preprocess(feature_df)
    shap_values = artifacts["explainer"](
        X_preprocessed, max_evals=2 * X_preprocessed.shape[1] + 1
    )
    tab1, tab2, tab3 = st.tabs(["Explanation", "Image", "Radiomics features"])
    with tab1:
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
