import os
from pathlib import Path

WEBAPP_DIR = Path(os.path.dirname(__file__))
TEMPLATE_DIR = WEBAPP_DIR / "templates"
TEMPLATE_DICT = {
    "Feature extraction": {
        "1. Dataset preparation": TEMPLATE_DIR
        / "feature_extraction"
        / "dataset_preparation.py",
        "2. Preprocessing": TEMPLATE_DIR
        / "feature_extraction"
        / "preprocessing.py",
        "3. Feature extraction": TEMPLATE_DIR
        / "feature_extraction"
        / "feature_extraction.py",
    },
    "Binary classification": {
        "1. Training": TEMPLATE_DIR / "binary_classification" / "training.py",
        "2. Evaluation": TEMPLATE_DIR
        / "binary_classification"
        / "evaluation.py",
        "3. Inference": TEMPLATE_DIR
        / "binary_classification"
        / "inference.py",
    },
    "Radiomics maps": {
        "Create maps": TEMPLATE_DIR / "radiomics_maps" / "create_maps.py",
    },
    "Segmentation": {
        "Generate code for nnUNet": TEMPLATE_DIR
        / "segmentation"
        / "generate_code_for_nnunet.py",
    },
}

IS_DEMO = False
if "INPUT_DIR" not in os.environ:
    IS_DEMO = True
