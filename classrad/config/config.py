import json
import logging
import os
import tempfile

import classrad

CONFIG_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(classrad.__file__)),
    "tests",
    "testing_data",
)
if "INPUT_DIR" in os.environ:
    INPUT_DIR = os.environ["INPUT_DIR"]
else:
    INPUT_DIR = tempfile.mkdtemp()
if "RESULT_DIR" in os.environ:
    RESULT_DIR = os.environ["RESULT_DIR"]
else:
    RESULT_DIR = tempfile.mkdtemp()

MODEL_REGISTRY = os.path.join(RESULT_DIR, "models")
# os.makedirs(MODEL_REGISTRY, exist_ok=True)
# mlflow.set_tracking_uri("file://" + MODEL_REGISTRY)

PARAM_DIR = os.path.join(CONFIG_DIR, "pyradiomics_params")
PRESETS = {
    "CT default": "default_feature_map.yaml",
    "CT reproducibility (Baessler et al.)": "Baessler_CT.yaml",
}
with open(os.path.join(CONFIG_DIR, "pyradiomics_feature_names.json")) as f:
    PYRADIOMICS_FEATURE_NAMES = json.load(f)

AVAILABLE_CLASSIFIERS = [
    "Random Forest",
    "Logistic Regression",
    "SVM",
    "XGBoost",
]
FEATURE_SELECTION_METHODS = ["anova", "lasso", "boruta"]
OVERSAMPLING_METHODS = ["SMOTE", "ADASYN", None]

SEED = 123

MONAI_DATA_DIR = tempfile.mkdtemp()

IS_DEMO = False

# Logging
logging.getLogger("classrad").addHandler(logging.NullHandler())
