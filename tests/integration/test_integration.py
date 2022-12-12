import os
from pathlib import Path

import pandas as pd
import pytest

from autorad.data.dataset import FeatureDataset, ImageDataset
from autorad.external.download_WORC import download_WORCDatabase
from autorad.feature_extraction.extractor import FeatureExtractor
from autorad.inference import infer, infer_utils
from autorad.models.classifier import MLClassifier
from autorad.preprocessing import preprocess
from autorad.training.trainer import Trainer
from autorad.utils.preprocessing import get_paths_with_separate_folder_per_case


@pytest.mark.parametrize(
    "use_feature_selection, preprocessing_kwargs, models",
    [
        (True, {"feature_selection_methods": None}, ["Random Forest"]),
        (True, {"feature_selection_methods": ["lasso"]}, ["XGBoost"]),
        (True, {"feature_selection_methods": ["lasso"]}, ["SVM"]),
        (
            True,
            {"feature_selection_methods": ["boruta", "anova"]},
            ["Random Forest", "XGBoost", "SVM", "Logistic Regression"],
        ),
        (False, {}, ["XGBoost"]),
    ],
)
# @pytest.mark.skip(reason="Slow")
def test_binary_classification(
    use_feature_selection, preprocessing_kwargs, models
):
    data_dir = Path(os.getenv("AUTORAD_INPUT_DIR"))
    base_dir = data_dir.parent
    # base_dir = Path(config.TEST_DATA_DIR) / "test_dataset"
    # data_dir = base_dir / "data"
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
    feature_df_path = result_dir / "features.csv"
    if not feature_df_path.is_file():
        feature_df = extractor.run()
        feature_df.to_csv(result_dir / "features.csv", index=False)
    else:
        feature_df = pd.read_csv(feature_df_path)

    label_df = pd.read_csv(data_dir / "labels.csv")
    merged_feature_df = feature_df.merge(
        label_df, left_on="ID", right_on="patient_ID", how="left"
    )
    feature_dataset = FeatureDataset(
        merged_feature_df, target="diagnosis", ID_colname="ID"
    )
    splits_path = result_dir / "splits.json"
    feature_dataset.split(method="train_val_test", save_path=splits_path)

    preprocess.run_auto_preprocessing(
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
    trainer.set_optimizer("optuna", n_trials=30)
    trainer.run(auto_preprocess=True)

    artifacts = infer_utils.get_artifacts_from_best_run()

    inferrer = infer.Inferrer(
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"],
        result_dir=result_dir,
    )
    # feature_df = infer.infer_radiomics_features(
    #     img_path,
    #     mask_path,
    #     artifacts["extraction_param_path"],
    # )
    feature_df.to_csv(Path(result_dir) / "infer_df.csv")
    result = inferrer.predict(feature_df)
    assert result is not None
