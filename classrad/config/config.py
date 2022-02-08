import json
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
MLFLOW_DIR = os.path.join(RESULT_DIR, "mlflow")

PARAM_DIR = os.path.join(CONFIG_DIR, "pyradiomics_params")
PRESETS = {
    "CT default": "default_feature_map.yaml",
    "CT reproducibility (Baessler et al.)": "Baessler_CT.yaml",
}
with open(os.path.join(CONFIG_DIR, "pyradiomics_feature_names.json")) as f:
    PYRADIOMICS_FEATURE_NAMES = json.load(f)

AVAILABLE_CLASSIFIERS = [
    "Random Forest",
    "AdaBoost",
    "Logistic Regression",
    "SVM",
    "Gaussian Process Classifier",
    "XGBoost",
]

SEED = 1234

MONAI_DATA_DIR = tempfile.mkdtemp()

IS_DEMO = False
