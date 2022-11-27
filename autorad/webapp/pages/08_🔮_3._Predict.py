from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import shap
import streamlit as st

from autorad.data.dataset import ImageDataset
from autorad.feature_extraction import extraction_utils
from autorad.models.classifier import MLClassifier
from autorad.training import train_utils
from autorad.training.infer import Inferrer
from autorad.visualization import plot_volumes
from autorad.webapp import template_utils, utils
from autorad.webapp.extractor import StreamlitFeatureExtractor


def get_best_run(experiment_id):
    all_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.AUC"]
    )
    try:
        best_run = all_runs.iloc[-1]
    except IndexError:
        st.error("No trained models found. Please run the training first.")
        st.stop()
    return best_run


def get_experiment_by_name(experiment_name="radiomics_binary"):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        st.error(
            "No experiment named 'radiomics_binary' found. "
            "Please run the training first."
        )
    return experiment_id


def get_params(run):
    params = [param for param in run.items() if param[0].startswith("param")]
    return params


def get_artifacts(run):
    uri = run["artifact_uri"]
    model = MLClassifier.load_from_mlflow(f"{uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{uri}/preprocessor")
    explainer = mlflow.shap.load_explainer(f"{uri}/shap-explainer")
    return model, preprocessor, explainer


def infer_radiomics_features(img_path, mask_path):
    path_df = pd.DataFrame(
        {
            "image_path": [img_path],
            "segmentation_path": [mask_path],
        }
    )
    image_dataset = ImageDataset(
        path_df,
        image_colname="image_path",
        mask_colname="segmentation_path",
    )
    extractor = StreamlitFeatureExtractor(
        image_dataset,
        extraction_params="MR_default.yaml",
    )
    feature_df = extractor.run()
    return feature_df


def show():
    template_utils.show_title()
    input_dir = utils.get_input_dir()
    result_dir = utils.get_result_dir()
    st.subheader("Select model")
    start_mlflow = st.button("Browse trained models in MLFlow")
    if start_mlflow:
        train_utils.start_mlflow_server()

    experiment_id = get_experiment_by_name("radiomics_binary")
    best_run = get_best_run(experiment_id)
    model, preprocessor, explainer = get_artifacts(best_run)

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
    inferrer = Inferrer(
        model=model,
        preprocessor=preprocessor,
        result_dir=result_dir,
    )
    with st.spinner("Extracting radiomics features..."):
        feature_df = infer_radiomics_features(img_path, mask_path)
    radiomics_features = extraction_utils.filter_pyradiomics_names(
        list(feature_df.columns)
    )
    feature_df = feature_df[radiomics_features]
    feature_df.to_csv(Path(result_dir) / "infer_df.csv")
    with st.spinner("Predicting..."):
        result = inferrer.predict(feature_df)
    st.write(f"For provided case probability of class=1 is: {result[0]:.2f}")
    shap.initjs()
    X_preprocessed = inferrer.preprocess(feature_df)
    shap_values = explainer(X_preprocessed)
    tab1, tab2, tab3 = st.tabs(["Explanation", "Image", "Radiomics features"])
    with tab1:
        fig = shap.plots.waterfall(shap_values[0])
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot(fig)
    with tab2:
        try:
            fig = plot_volumes.plot_roi(img_path, mask_path)
            fig.update_layout(width=200, height=200)
            st.plotly_chart(fig)
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
