<p align="center">
<br>
  <img src="docs/images/logo.png" alt="AutoRadiomics">
</p>

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/pwoznicki/AutoRadiomics/actions/workflows/testing.yml/badge.svg)](https://github.com/pwoznicki/AutoRadiomics/commits/main)
[![codecov](https://codecov.io/gh/pwoznicki/AutoRadiomics/branch/main/graph/badge.svg)](https://codecov.io/gh/pwoznicki/AutoRadiomics)

## Simple pipeline for experimenting with radiomics features

| <p align="center"><a href="https://share.streamlit.io/pwoznicki/autoradiomics/main/webapp/app.py"> Streamlit Share | <p align="center"><a href="https://hub.docker.com/repository/docker/piotrekwoznicki/classy-radiomics"> Docker | <p align="center"><a href="https://pypi.org/project/classrad/"> Python                                         |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/streamlit.png" /></p>  | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/docker.png"/></p> | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/python.png" /></p> |
| <p align="center"><a href="https://share.streamlit.io/pwoznicki/autoradiomics/main/webapp/app.py"> **Demo**        | `docker run -p 8501:8501 -v <your_data_dir>:/data -it piotrekwoznicki/classy-radiomics:0.1`                   | `pip install --upgrade classrad`                                                                               |

&nbsp;

### Installation from source

```bash
git clone https://github.com/pwoznicki/AutoRadiomics.git
cd AutoRadiomics
pip install -e .
```

## Example - Hydronephrosis detection from CT images:

### Extract radiomics features

```python
df = pd.read_csv(table_dir / "paths.csv")
image_dataset = ImageDataset(
    df=df,
    image_colname="image path",
    mask_colname="mask path",
    ID_colname="patient ID"
)
extractor = FeatureExtractor(
    dataset=image_dataset,
    out_path=(table_dir / "features.csv"),
)
extractor.extract_features()
```

### Load, split and preprocess extracted features

```python
# Create a dataset from the radiomics features
feature_df = pd.read_csv(table_dir / "features.csv")
feature_dataset = FeatureDataset(
    dataframe=feature_df,
    target="Hydronephrosis",
    task_name="Hydronephrosis detection"
)

# Split data and load splits
splits_path = result_dir / "splits.json"
feature_dataset.full_split(save_path=splits_path)
feature_dataset.load_splits_from_json(splits_path)

# Preprocessing
preprocessor = Preprocessor(
    normalize=True,
    feature_selection_method="boruta",
    oversampling_method="SMOTE",
)
feature_dataset._data = preprocessor.fit_transform(dataset.data)
```

### Train the model for hydronephrosis classification

```python
# Select classifiers to compare
classifier_names = [
    "Gaussian Process Classifier",
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "XGBoost",
]
classifiers = [MLClassifier.from_sklearn(name) for name in classifier_names]

model = MLClassifier.from_sklearn(name="Random Forest")
model.set_optimizer("optuna", n_trials=5)

trainer = Trainer(
    dataset=dataset,
    models=[model],
    result_dir=result_dir,
    experiment_name="Hydronephrosis detection"
)
trainer.run()
```

### Create an evaluator to train and evaluate selected classifiers

```python
evaluator = Evaluator(dataset=data, models=classifiers)
evaluator.evaluate_cross_validation()
evaluator.boxplot_by_class()
evaluator.plot_all_cross_validation()
evaluator.plot_test()
```

## Commands

### MLFlow

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri <result_dir>
```

## Dependencies:

- MONAI
- pyRadiomics
- MLFlow
- Optuna
- scikit-learn
- imbalanced-learn
- XGBoost
- Boruta
- Medpy
- NiBabel
- SimpleITK
- nilearn
- LOFO-importance
- plotly
- seaborn

### App dependencies:

- Streamlit
- Docker
