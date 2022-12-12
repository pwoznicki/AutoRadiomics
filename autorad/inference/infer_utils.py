import mlflow

from autorad.models.classifier import MLClassifier
from autorad.utils import io, mlflow_utils


def get_artifacts_from_best_run(experiment_name="model_training"):
    experiment_id = mlflow_utils.get_experiment_id_from_name(experiment_name)
    best_run = mlflow_utils.get_best_run(experiment_id)
    artifacts = get_artifacts(best_run)
    return artifacts


def get_artifacts(run):
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
