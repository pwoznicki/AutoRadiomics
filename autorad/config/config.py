import json
import logging
import logging.config
import os
import sys
import tempfile
from pathlib import Path

from rich.logging import RichHandler

import autorad

CONFIG_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(autorad.__file__)),
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
FEATURE_SELECTION_METHODS = ["anova", "lasso", "boruta", "boruta-shap"]
OVERSAMPLING_METHODS = ["SMOTE", "ADASYN", None]

SEED = 123

MONAI_DATA_DIR = tempfile.mkdtemp()

IS_DEMO = False

# Logging
LOGS_DIR = Path(RESULT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
