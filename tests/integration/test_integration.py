from pathlib import Path

import pandas as pd
import pytest

from autorad.config import config
from autorad.data.dataset import FeatureDataset, ImageDataset
from autorad.external.download_WORC import download_WORCDatabase
from autorad.feature_extraction.extractor import FeatureExtractor
from autorad.models.classifier import MLClassifier
from autorad.training.infer import Inferrer
from autorad.training.trainer import Trainer
from autorad.utils import io
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case
from autorad.visualization import plotly_utils


@pytest.mark.parametrize(
    "feature_selection, selection_methods",
    [(True, "all"), (True, ["lasso"]), (False, None)],
)
def test_binary_classification(feature_selection, selection_methods):
    base_dir = Path(config.TEST_DATA_DIR) / "test_dataset"
    data_dir = base_dir / "data"
    result_dir = base_dir / "results"
    data_dir.mkdir(exist_ok=True, parents=True)
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
    extractor = FeatureExtractor(
        image_dataset, extraction_params="MR_default.yaml"
    )
    feature_df = extractor.run()

    label_df = pd.read_csv(data_dir / "labels.csv")
    merged_feature_df = feature_df.merge(
        label_df, left_on="ID", right_on="patient_ID", how="left"
    )
    feature_dataset = FeatureDataset(
        merged_feature_df, target="diagnosis", ID_colname="ID"
    )
    splits_path = result_dir / "splits.json"
    feature_dataset.split(method="train_val_test", save_path=splits_path)

    models = MLClassifier.initialize_default_sklearn_models()
    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir=result_dir,
        experiment_name="Fibromatosis_vs_sarcoma_classification",
    )
    trainer.run_auto_preprocessing(
        feature_selection=feature_selection,
        selection_methods=selection_methods,
        oversampling=False,
    )

    trainer.set_optimizer("optuna", n_trials=30)
    trainer.run(auto_preprocess=True)

    best_params = io.load_json(result_dir / "best_params.json")
    inferrer = Inferrer(params=best_params, result_dir=result_dir)
    inferrer.fit_eval(feature_dataset, result_name="test")

    results = pd.read_csv(result_dir / "test.csv")
    plotly_utils.plot_roc_curve(results.y_true, results.y_pred_proba)
