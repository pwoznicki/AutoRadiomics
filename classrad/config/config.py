import os
import json

AVAILABLE_CLASSIFIERS = [
    "Random Forest",
    "AdaBoost",
    "Logistic Regression",
    "SVM",
    "Gaussian Process Classifier",
    "XGBoost",
]

SEED = 1234

PRESETS = {"default CT": "default_feature_map.yaml"}
CONFIG_DIR = os.path.dirname(__file__)
PARAM_DIR = os.path.join(CONFIG_DIR, "pyradiomics_params")

with open(os.path.join(CONFIG_DIR, "pyradiomics_feature_names.json")) as f:
    PYRADIOMICS_FEATURE_NAMES = json.load(f)
