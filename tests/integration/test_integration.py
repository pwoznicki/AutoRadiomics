from pathlib import Path
import shutil
import uuid

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from autorad.config import config
from autorad.data import FeatureDataset, ImageDataset
from autorad import evaluation
from autorad.external.download_WORC import download_WORCDatabase
from autorad.feature_extraction import FeatureExtractor
from autorad.inference import infer_utils
from autorad.models.classifier import MLClassifier
from autorad.preprocessing import run_auto_preprocessing
from autorad.training import Trainer
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case


@pytest.mark.parametrize(
    "use_feature_selection, preprocessing_kwargs, split_method, models",
    [
        (True, {"feature_selection_methods": None}, "train_val_test", ["Random Forest"]),
        (True, {"feature_selection_methods": ["lasso"]}, "train_with_cross_validation_test", ["XGBoost"]),
        (True, {"feature_selection_methods": ["lasso"]}, "train_val_test", ["SVM"]),
        (
            True,
            {"feature_selection_methods": [None, "boruta", "anova"]}, "train_with_cross_validation_test",
            ["Random Forest", "XGBoost", "SVM", "Logistic Regression"],
        ),
        (False, {}, "train_val_test", ["XGBoost"]),
    ],
)
@pytest.mark.slow
def test_binary_classification(
    use_feature_selection, preprocessing_kwargs, split_method, models
):
    base_dir = Path(config.TEST_DATA_DIR) / "test_dataset"
    data_dir = base_dir / "data"
    result_dir = base_dir / "results"
    data_dir.mkdir(exist_ok=True, parents=True)
    if result_dir.is_dir():
        shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(exist_ok=True, parents=True)


    dataset_name = "Desmoid"
    if not list(data_dir.iterdir()):
        download_WORCDatabase(
            dataset=dataset_name,
            data_folder=data_dir,
            n_subjects=100,
        )

    if not len(list(data_dir.glob(f"*{dataset_name}*"))) == 100:
        raise ValueError("Downloaded dataset is incomplete.")

    path_df = get_paths_with_separate_folder_per_case(data_dir, relative=True)

    image_dataset = ImageDataset(
        path_df,
        ID_colname="ID",
        root_dir=data_dir,
    )
    feature_df_path = result_dir / "features.csv"
    if not feature_df_path.is_file():
        extractor = FeatureExtractor(
            image_dataset, extraction_params="MR_default.yaml", n_jobs=-1
        )
        feature_df = extractor.run()
        feature_df.to_csv(feature_df_path, index=False)
    else:
        feature_df = pd.read_csv(feature_df_path)

    label_df = pd.read_csv(data_dir / "labels.csv")
    merged_feature_df = feature_df.merge(
        label_df, left_on="ID", right_on="patient_ID", how="left"
    )
    feature_dataset = FeatureDataset(
        merged_feature_df, target="diagnosis", ID_colname="ID"
    )
    splits_path = result_dir / "splits.yaml"
    feature_dataset.split(method=split_method, save_path=splits_path)

    run_auto_preprocessing(
        data=feature_dataset.data,
        result_dir=result_dir,
        use_feature_selection=use_feature_selection,
        use_oversampling=False,
        **preprocessing_kwargs,
    )

    model_objects = [MLClassifier.from_sklearn(model) for model in models]
    trainer = Trainer(
        dataset=feature_dataset,
        models=model_objects,
        result_dir=result_dir,
    )
    trainer.set_optimizer("optuna", n_trials=300)
    experiment_name = "model_training" + str(uuid.uuid4())
    trainer.run(auto_preprocess=True, experiment_name=experiment_name)

    best_run = infer_utils.get_best_run_from_experiment_name(experiment_name)
    artifacts = infer_utils.load_pipeline_artifacts(best_run)
    result_df = evaluation.evaluate_feature_dataset(
        dataset=feature_dataset,
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"],
    )
    result_df.to_csv(result_dir / "predictions.csv", index=False)

    assert result_df is not None

    # assert there's a correlation between 'y_pred' and 'y_true' in the result_df
    assert (
        result_df["y_pred_proba"].astype(float).corr(result_df["y_true"]) > 0
    )

    # assert AUC is higher than 0.5
    y_pred = result_df["y_pred_proba"] > 0.5
    assert roc_auc_score(result_df["y_true"], y_pred) > 0.5
