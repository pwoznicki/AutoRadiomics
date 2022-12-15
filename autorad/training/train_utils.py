import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import shap

from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from autorad.utils import io, mlflow_utils


def get_model_by_name(name, models):
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model with name {name} not found")


def log_splits(splits: dict):
    mlflow_utils.log_dict_as_artifact(splits, "splits")


def log_shap(model: MLClassifier, X_train: pd.DataFrame):
    explainer = shap.Explainer(model.predict_proba_binary, X_train)
    mlflow.shap.log_explainer(explainer, "shap-explainer")


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)


def log_dataset(dataset: FeatureDataset):
    dataset_config = {
        "target": dataset.target,
        "ID_colname": dataset.ID_colname,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / "feature_dataset"
        save_dir.mkdir(exist_ok=True)
        io.save_yaml(dataset_config, save_dir / "dataset_config.yaml")
        dataset.df.to_csv(save_dir / "df.csv", index=False)
        mlflow.log_artifacts(str(save_dir), "feature_dataset")
