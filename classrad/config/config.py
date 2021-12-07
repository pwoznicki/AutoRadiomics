import json
import os

import classrad

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
TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(classrad.__file__)),
    "tests",
    "testing_data",
)

with open(os.path.join(CONFIG_DIR, "pyradiomics_feature_names.json")) as f:
    PYRADIOMICS_FEATURE_NAMES = json.load(f)
