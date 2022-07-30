import subprocess
import webbrowser
from pathlib import Path

import seaborn as sns
import streamlit as st

from autorad.config import config
from autorad.data.dataset import FeatureDataset
from autorad.models.classifier import MLClassifier
from autorad.training.trainer import Trainer
from autorad.webapp import utils


def show():
    with st.sidebar:
        st.write(
            """
            Given a table with radiomics features extracted in previous step,
            run the training consisting of:
            - Feature selection
            - Hyperparameter optimization for selected models
        """
        )
    feature_df = utils.load_df("Choose a CSV file with radiomics features")
    cm = sns.light_palette("green", as_cmap=True)
    st.dataframe(feature_df.style.background_gradient(cmap=cm))
    all_colnames = feature_df.columns.tolist()
    with st.form("Training config"):
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select the label", all_colnames)
        with col2:
            ID_colname = st.selectbox("Select patient ID", all_colnames)
        feature_dataset = FeatureDataset(
            dataframe=feature_df,
            target=target,
            ID_colname=ID_colname,
        )
        col1, col2 = st.columns(2)
        with col1:
            split_method = st.radio(
                label="Splitting",
                options=["train_val_test", "train_with_cross_validation_test"],
            )
        with col2:
            test_size_percent = st.slider(
                label="Test size (%)",
                min_value=0,
                max_value=100,
                value=20,
            )
        test_size = test_size_percent / 100
        model_names = st.multiselect(
            "Select the models", config.AVAILABLE_CLASSIFIERS
        )

        start_training = st.form_submit_button("Start training")
        if start_training:
            run_training_mlflow(
                feature_dataset, split_method, test_size, model_names
            )
    if st.button("Track the models with MLflow dashboard"):
        mlflow_model_dir = (Path(config.RESULT_DIR) / "models").absolute()
        subprocess.Popen(
            [
                "mlflow",
                "server",
                "-h",
                "0.0.0.0",
                "-p",
                "8000",
                "--backend-store-uri",
                mlflow_model_dir,
            ]
        )
        webbrowser.open("http://localhost:8000/")


def run_training_mlflow(feature_dataset, split_method, test_size, model_names):
    result_dir = config.RESULT_DIR

    feature_dataset.split(
        method=split_method,
        test_size=test_size,
        save_path=(Path(result_dir) / "splits.json"),
    )

    models = [MLClassifier.from_sklearn(name) for name in model_names]
    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir=result_dir,
    )
    with st.spinner("Preprocessing in progress..."):
        trainer.run_auto_preprocessing(
            selection_methods=["boruta"], oversampling=False
        )
    trainer.set_optimizer("optuna", n_trials=30)
    with st.spinner("Training in progress..."):
        trainer.run(auto_preprocess=False)
    st.success(
        f"Training done! Predictions saved in your result directory \
        ({result_dir})"
    )


if __name__ == "__main__":
    show()
