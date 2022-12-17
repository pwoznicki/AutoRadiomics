import mlflow
import pandas as pd

from autorad.data import FeatureDataset
from autorad.models.classifier import MLClassifier
from autorad.utils import io, mlflow_utils


def get_best_run_from_experiment_name(experiment_name):
    experiment_id = mlflow_utils.get_experiment_id_from_name(experiment_name)
    best_run = mlflow_utils.get_best_run(experiment_id)

    return best_run


def load_pipeline_artifacts(run):
    uri = run["artifact_uri"]
    model = MLClassifier.load_from_mlflow(f"{uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{uri}/preprocessor")
    explainer = mlflow.shap.load_explainer(f"{uri}/shap-explainer")
    extraction_config = io.load_yaml(
        f"{uri.removeprefix('file://')}/feature_extraction/extraction_config.yaml"
    )
    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "extraction_config": extraction_config,
    }
    return artifacts


def load_dataset_artifacts(run):
    uri = run["artifact_uri"]
    splits = io.load_yaml(f"{uri.removeprefix('file://')}/splits.yaml")
    df = pd.read_csv(f"{uri.removeprefix('file://')}/feature_dataset/df.csv")
    dataset_config = io.load_yaml(
        f"{uri.removeprefix('file://')}/feature_dataset/dataset_config.yaml"
    )
    artifacts = {
        "df": df,
        "dataset_config": dataset_config,
        "splits": splits,
    }
    return artifacts


def load_feature_dataset(feature_df, dataset_config, splits) -> FeatureDataset:
    dataset = FeatureDataset(
        dataframe=feature_df,
        target=dataset_config["target"],
        ID_colname=dataset_config["ID_colname"],
    )
    dataset.load_splits(splits)

    return dataset
