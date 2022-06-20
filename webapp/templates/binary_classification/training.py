from dataclasses import dataclass

import pandas as pd
import seaborn as sns
import streamlit as st
import utils

from autorad.config import config
from autorad.data.dataset import FeatureDataset
from autorad.models.classifier import MLClassifier
from autorad.training.trainer import Trainer


@dataclass
class TrainingConfig:
    feature_df: pd.DataFrame = feature_df
    label_colname: str | None = None
    ID_colname: str | None = None
    model_names: list[str] | None = None


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
    trainer_config = TrainingConfig(feature_df)

    col1, col2 = st.columns(2)
    with col1:
        label = st.selectbox("Select the label", trainer_config.all_colnames())
    with col2:
        pat_id = st.selectbox(
            "Select patient ID", trainer_config.all_colnames()
        )
    model_names = st.multiselect(
        "Select the models", config.AVAILABLE_CLASSIFIERS
    )
    num_features = st.slider(
        "Number of features", min_value=2, max_value=50, value=10
    )

    trainer_config.set_label_colname(label)
    trainer_config.set_ID_colname(pat_id)
    trainer_config.set_model_names(model_names)
    trainer_config.set_num_features(num_features)

    track_with_mlflow = st.checkbox("Track with mlflow?")

    start_training = st.button("Start training")
    if start_training:
        if track_with_mlflow:
            run_training_mlflow(trainer_config)
        # else:
        #    run_training_autorad(trainer_config)


def run_training_mlflow(trainer_config: TrainingConfig):
    data = FeatureDataset(
        dataframe=trainer_config.get_feature_df(),
        features=trainer_config.get_feature_names(),
        target=trainer_config.get_label_colname(),
        ID_colname=trainer_config.get_ID_colname(),
        task_name="Task_placeholder",
    )
    data.full_split()

    models = [MLClassifier(name) for name in trainer_config.get_model_names()]
    result_dir = config.RESULT_DIR
    trainer = Trainer(
        dataset=data,
        models=models,
        result_dir=result_dir,
        num_features=trainer_config.get_num_features(),
        meta_colnames=[
            trainer_config.get_ID_colname(),
            trainer_config.get_label_colname(),
        ],
    )
    with st.spinner("Training in progress..."):
        trainer.train_cross_validation()
    fig = trainer.dataset.boxplot_by_class()
    st.success(
        f"Training done! Predictions saved in your result directory \
        ({result_dir})"
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    show()
