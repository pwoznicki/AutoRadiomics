from pathlib import Path

import mlflow
import streamlit as st
import utils
from classrad.config.config import Config
from classrad.data.dataset import Dataset
from classrad.models.classifier import MLClassifier
from classrad.training.trainer import Trainer

result_dir = Path("/Users/p.woznicki/Documents/test")


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    with st.sidebar:
        pass
    config = Config()
    feature_df = utils.load_df("Choose a CSV file with radiomics features")
    st.write(feature_df)
    feature_df.dropna(axis="index", inplace=True)
    feature_df_colnames = feature_df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        label = st.selectbox("Select the label", feature_df_colnames)
    with col2:
        pat_id = st.selectbox("Select patient ID", feature_df_colnames)

    available_classifiers = config.available_classifiers

    model_names = st.multiselect("Select the models", available_classifiers)
    num_features = st.slider("Number of features", min_value=2, max_value=50, value=10)
    # Mlflow tracking
    track_with_mlflow = st.checkbox("Track with mlflow?")

    # Model training
    start_training = st.button("Start training")
    if start_training:
        feature_colnames = [
            col
            for col in feature_df.columns
            if col.startswith(("original", "wavelet", "shape"))
        ]
        data = Dataset(feature_df, feature_colnames, label, "Harnstau")
        data.cross_validation_split_by_patient(patient_colname=pat_id)

        models = [MLClassifier(name) for name in model_names]

        trainer = Trainer(
            dataset=data,
            models=models,
            result_dir=result_dir,
            num_features=num_features,
            meta_colnames=[pat_id, label],
        )
        with st.spinner("Training in progress..."):
            trainer.train_cross_validation()
        fig = trainer.dataset.boxplot_by_class()
        st.success(
            f"Training done! Predictions saved in your result directory ({result_dir})"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    show()
