from pathlib import Path

import pandas as pd
import shap
import streamlit as st

from autorad.config import config
from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from autorad.preprocessing import preprocess
from autorad.training import Trainer
from autorad.utils import mlflow_utils
from autorad.webapp import st_read, st_utils


def merge_labels_with_features():
    with st.expander(
        "You have labels in another table? " "Merge them with features here"
    ):
        data_dir = st_read.get_input_dir()
        result_dir = st_read.get_result_dir()
        st.write(f"Put the table with labels here: {data_dir}")
        col1, col2 = st.columns(2)
        with col1:
            label_df_path = st_read.file_selector(
                data_dir, "Select the table with labels", "csv"
            )
            label_df = pd.read_csv(label_df_path)
        with col2:
            feature_df_path = st_read.file_selector(
                result_dir, "Select the table with radiomics features", "csv"
            )
            feature_df = pd.read_csv(feature_df_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            label = st.selectbox(
                "Select the label",
                label_df.columns,
                index=st_read.guess_idx_of_column(label_df.columns, "label"),
            )
        with col2:
            ID_label = st.selectbox(
                "Select patient ID for the label table",
                label_df.columns,
                index=st_read.guess_idx_of_column(label_df.columns, "id"),
            )
        with col3:
            ID_feature = st.selectbox(
                "Select patient ID for the feature table",
                feature_df.columns,
                index=st_read.guess_idx_of_column(label_df.columns, "id"),
            )
        save_path = st.text_input(
            "Where to save the merged table",
            f"{feature_df_path[:-4]}_merged.csv",
        )

        if st.button("Merge tables"):
            label_df = label_df[[ID_label, label]].rename(
                {ID_label: ID_feature},
                axis=1,
            )
            merged_df = label_df.merge(
                feature_df,
                how="right",
                on=ID_feature,
            )
            if merged_df[label].isna().any():
                st.error(
                    "Some labels are missing. It's recommended that you "
                    "manually add the labels."
                )
            else:
                st.success("Labels merged successfully!")
            merged_df.to_csv(save_path, index=False)


def show():
    st_utils.show_title()
    st.write(
        """
        Given a table with radiomics features extracted in previous step,
        run the training consisting of:
        - Feature selection
        - Hyperparameter optimization for classifiers
    """
    )
    st.info(
        """For training, you will need binary labels to train the model on.
        The labels need to be added to the .csv table with features."""
    )
    st.info(
        """
        If you have the labels in a separate table, you can use our merging tool below.
        Otherwise please do it manually and save the new table in your `result_dir`."""
    )

    merge_labels_with_features()
    result_dir = st_read.get_result_dir()
    feature_df_path = st_read.file_selector(
        result_dir,
        "Choose a CSV table with radiomics features:",
        ext="csv",
    )
    feature_df = pd.read_csv(feature_df_path)
    st.dataframe(feature_df)
    all_colnames = feature_df.columns.tolist()
    with st.form("Training config"):
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Select the label",
                all_colnames,
                index=st_read.guess_idx_of_column(all_colnames, "label"),
            )
        with col2:
            ID_colname = st.selectbox(
                "Select patient ID",
                all_colnames,
                index=st_read.guess_idx_of_column(all_colnames, "id"),
            )
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
        feature_selection_methods = st.multiselect(
            "Select feature selection methods",
            config.FEATURE_SELECTION_METHODS,
        )
        model_names = st.multiselect(
            "Select the models", config.AVAILABLE_CLASSIFIERS
        )
        n_trials = st.number_input(
            "Number of trials for hyperparameter optimization",
            min_value=1,
            value=50,
        )
        start_training = st.form_submit_button("Start training")
        if start_training:
            run_training_mlflow(
                feature_dataset=feature_dataset,
                split_method=split_method,
                test_size=test_size,
                feature_selection_methods=feature_selection_methods,
                model_names=model_names,
                n_trials=n_trials,
                result_dir=result_dir,
            )
    if st.button("Inspect the models with MLflow dashboard"):
        mlflow_utils.start_mlflow_server()
        mlflow_utils.open_mlflow_dashboard()

    st_utils.next_step("2.2_Evaluate")


def show_interpretability(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    fig = shap.plots.beesward(shap_values)
    st.pyplot(fig)


def run_training_mlflow(
    feature_dataset: FeatureDataset,
    split_method,
    test_size,
    feature_selection_methods,
    model_names,
    n_trials,
    result_dir,
):

    feature_dataset.split(
        method=split_method,
        test_size=test_size,
        save_path=(Path(result_dir) / "splits.yaml"),
    )
    with st.spinner("Preprocessing in progress..."):
        preprocess.run_auto_preprocessing(
            data=feature_dataset.data,
            result_dir=result_dir,
            feature_selection_methods=feature_selection_methods,
            use_oversampling=False,
        )
    models = [MLClassifier.from_sklearn(name) for name in model_names]
    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir=result_dir,
    )
    trainer.set_optimizer("optuna", n_trials=n_trials)
    with st.spinner("Training in progress..."):
        trainer.run(auto_preprocess=True)
    st.success(
        f"Training done! Predictions saved in your result directory \
        ({result_dir})"
    )


if __name__ == "__main__":
    show()
