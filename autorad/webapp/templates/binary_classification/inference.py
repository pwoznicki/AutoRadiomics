from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from autorad.data.dataset import ImageDataset
from autorad.feature_extraction import extraction_utils
from autorad.training import train_utils
from autorad.training.infer import Inferrer
from autorad.webapp import template_utils, utils
from autorad.webapp.extractor import StreamlitFeatureExtractor


def get_best_params_and_models():
    try:
        experiment_id = mlflow.get_experiment_by_name(
            "radiomics_binary"
        ).experiment_id
    except AttributeError:
        st.error(
            "No experiment named 'radiomics_binary' found. "
            "Please run the training first."
        )
    all_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.AUC"]
    )
    # Get the best run
    best_run = all_runs.iloc[-1]
    # Get the best model
    best_uri = best_run["artifact_uri"]
    best_model = mlflow.pyfunc.load_model(f"{best_uri}/model")
    best_preprocessor = mlflow.sklearn.load_model(f"{best_uri}/preprocessor")
    best_params = [
        param for param in best_run.items() if param[0].startswith("param")
    ]
    return best_params, best_model, best_preprocessor


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
        extraction_params="CT_test.yaml",
    )
    feature_df = extractor.run()
    return feature_df


def show():
    input_dir = utils.get_input_dir()
    result_dir = utils.get_result_dir()
    template_utils.find_all_data(input_dir)
    img_path, mask_path = template_utils.read_image_seg_paths()

    start_mlflow = st.button("Browse trained models")
    if start_mlflow:
        train_utils.start_mlflow_server()
    with st.form("run inference"):
        params, model, preprocessor = get_best_params_and_models()
        run_prediction = st.form_submit_button("Predict")
        if run_prediction:
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
            st.write(f"For given case the predicted class is: {result}")


if __name__ == "__main__":
    show()
