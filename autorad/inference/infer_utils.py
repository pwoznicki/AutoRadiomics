import mlflow

from autorad.models.classifier import MLClassifier


def get_best_run(experiment_id):
    all_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.AUC"]
    )
    try:
        best_run = all_runs.iloc[-1]
    except IndexError:
        raise IndexError(
            "No trained models found. Please run the training first."
        )
    return best_run


def get_experiment_by_name(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"No experiment named {experiment_name} found. "
            "Please run the training first."
        )
    experiment_id = experiment.experiment_id
    return experiment_id


def get_artifacts(run):
    uri = run["artifact_uri"]
    model = MLClassifier.load_from_mlflow(f"{uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{uri}/preprocessor")
    explainer = mlflow.shap.load_explainer(f"{uri}/shap-explainer")
    extraction_param_path = (
        f"{uri.removeprefix('file://')}/extraction_params.json"
    )
    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "extraction_param_path": extraction_param_path,
    }
    return artifacts


def get_artifacts_from_best_run(experiment_name="model_training"):
    experiment_id = get_experiment_by_name(experiment_name)
    best_run = get_best_run(experiment_id)
    artifacts = get_artifacts(best_run)
    return artifacts
