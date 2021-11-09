import streamlit as st
import mlflow
from pathlib import Path
import pandas as pd
import collections
import os
from Radiomics.evaluation.evaluator import Evaluator
from Radiomics.models.classifier import MLClassifier
from Radiomics.config.config import Config
from Radiomics.data.dataset import Dataset
from Radiomics.training.trainer import Trainer

import utils

# input: result_df
# output: plots


# @st.cache(persist=False)
def load_df(label):
    uploaded_file = st.file_uploader(label)
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)
    return df


def main():
    # SET UP AVAILABLE TEMPLATES
    template_dict = collections.defaultdict(dict)
    template_dirs = [f for f in os.scandir("templates") if f.is_dir()]
    template_dirs = sorted(template_dirs, key=lambda e: e.name)
    for template_dir in template_dirs:
        try:
            # Templates with task + framework.
            task, framework = template_dir.name.split("_")
            template_dict[task][framework] = template_dir.path
        except ValueError:
            # Templates with task only.
            template_dict[template_dir.name] = template_dir.path
    print(template_dict)

    st.set_page_config(layout="wide")

    col1, col2 = st.columns(2)
    with col1:
        st.title("Radiomics analysis")
    with col2:
        st.write(
            """
        #### Simple pipeline for modelling using `pyradiomics` and `scikit-learn`.
        """
        )
    config = Config()

    with st.sidebar:
        st.write("## Task")
        task = st.selectbox("Select workflow", list(template_dict.keys()))
        if isinstance(template_dict[task], dict):
            framework = st.selectbox(f"Select task", list(template_dict[task].keys()))
            template_dir = template_dict[task][framework]
        else:
            template_dir = template_dict[task]

    # Show template-specific sidebar components (based on sidebar.py in the template dir).
    template_sidebar = utils.import_from_file(
        "template_sidebar", os.path.join(template_dir, "sidebar.py")
    )
    inputs = template_sidebar.show()

    feature_df = load_df("Choose a CSV file with radiomics features")
    # feature_df = pd.read_csv(uploaded_file)
    feature_df.dropna(axis="index", inplace=True)
    feature_df_colnames = feature_df.columns.tolist()
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        label = st.selectbox("Select the label", feature_df_colnames)
    with col1_2:
        pat_id = st.selectbox("Select patient ID", feature_df_colnames)

    available_classifiers = config.available_classifiers

    model_names = st.multiselect("Select the models", available_classifiers)
    num_features = st.slider("Number of features", min_value=2, max_value=50, value=10)
    # Mlflow tracking
    track_with_mlflow = st.checkbox("Track with mlflow?")

    result_dir = Path("/Users/p.woznicki/Documents/test")

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
            meta_colnames=[pat_id],
        )
        with st.spinner("Wait for it..."):
            trainer.train_cross_validation()
        fig = trainer.dataset.boxplot_by_class()
        st.success("Training done!")
        st.plotly_chart(fig, use_container_width=True)

    # LAYING OUT THE TOP SECTION OF THE APP
    col1, col2 = st.columns((2, 3))

    with col1:
        result_df = load_df("Choose a CSV file with results")
    with col2:
        st.write("#### Now let us analyze the results.")

    # Evaluation
    evaluate = st.button("Evaluate!")
    if evaluate:
        evaluator = Evaluator(
            result_df=result_df,
            target="patient_label",
            result_dir=result_dir,
        )
        evaluator.evaluate()
        st.write(evaluator.plot_roc_curve_all())
        st.write(evaluator.plot_confusion_matrix_all())
        st.write(evaluator.plot_test())

    # LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
    # row2_1, row2_2, row2_3, row2_4 = st.columns((2,1,1,1))


if __name__ == "__main__":
    main()
