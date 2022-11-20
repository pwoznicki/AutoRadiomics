import mlflow
import pandas as pd
import streamlit as st

from autorad.data.dataset import ImageDataset
from autorad.feature_extraction.extractor import FeatureExtractor


def get_best_model():
    # Load all runs from experiment
    experiment_id = mlflow.get_experiment_by_name("baseline").experiment_id
    all_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.AUC"]
    )
    # Get the best run
    best_run = all_runs.iloc[0]
    # Get the best model
    best_model = mlflow.pyfunc.load_model(best_run["artifact_uri"])
    return best_model


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
    extractor = FeatureExtractor(
        image_dataset,
        extraction_params="MR_default.yaml",
    )
    feature_df = extractor.run()
    return feature_df


def show():
    # model = mlflow.pyfunc.load_model(model_path)
    col1, col2 = st.columns(2)
    with col1:
        img_path = st.file_uploader(label="Input image")
    with col2:
        mask_path = st.file_uploader(label="Input segmentation")
    model = get_best_model()
    if st.button("Predict"):
        if not img_path or not mask_path:
            st.error("Image or segmentation not found.")
        feature_df = infer_radiomics_features(img_path, mask_path)
        result = model.predict(feature_df)
        st.write(result)


if __name__ == "__main__":
    show()
