import streamlit as st
import utils
from classrad.config import config
from classrad.data.dataset import Dataset
from classrad.models.classifier import MLClassifier
from classrad.training.trainer import Trainer


class TrainingConfig:
    def __init__(self, feature_df):
        self.feature_df = feature_df
        self.label_colname = None
        self.ID_colname = None
        self.model_names = None
        self.num_features = None

    def set_label_colname(self, label_colname):
        self.label_colname = label_colname

    def set_ID_colname(self, ID_colname):
        self.ID_colname = ID_colname

    def set_model_names(self, model_names):
        self.model_names = model_names

    def set_num_features(self, num_features):
        self.num_features = num_features

    def get_feature_df(self):
        return self.feature_df

    def all_colnames(self):
        return self.feature_df.columns.tolist()

    def get_feature_names(self):
        assert self.feature_df is not None, "Feature dataframe not set!"
        feature_names = [
            col
            for col in self.feature_df.columns.tolist()
            if col.startswith(("original", "wavelet", "shape"))
        ]
        return feature_names

    def get_label_colname(self):
        return self.label_colname

    def get_ID_colname(self):
        return self.ID_colname

    def get_model_names(self):
        return self.model_names

    def get_num_features(self):
        return self.num_features


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
    st.write(feature_df)
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
        #    run_training_classrad(trainer_config)


def run_training_mlflow(trainer_config: TrainingConfig):
    data = Dataset(
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
