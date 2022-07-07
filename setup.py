import numpy  # noqa: F401
import setuptools
from setuptools import setup

with open("requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "coverage==6.2",
    "great-expectations==0.14.2",
    "pytest==6.2.5",
    "pytest-watch",
    "hypothesis==6.36.0",
]

dev_packages = [
    "black==21.12b0",
    "flake8==4.0.1",
    "isort==5.10.1",
    "pre-commit==2.17.0",
]

webapp_packages = ["streamlit==1.10.0", "docker==5.0.3", "jupytext==1.13.8"]

docs_packages = [
    "mkdocs==1.3.0",
    "mkdocs-material==8.3.3",
    "mkdocstrings==0.18.1",
    "mkdocstrings-python-legacy",
]

setup(
    name="autorad",
    packages=setuptools.find_packages(),
    package_data={
        "autorad": [
            "config/pyradiomics_feature_names.json",
            "webapp/paths_example.csv",
            "webapp/templates/segmentation/pretrained_models.json",
            "config/pyradiomics_params/*",
        ]
    },
    setup_requires=["numpy"],
    include_package_data=True,
    install_requires=required_packages,
    extras_require={
        "app": webapp_packages,
        "dev": test_packages + dev_packages + webapp_packages + docs_packages,
        "docs": docs_packages,
    },
    entry_points={
        "console_scripts": [
            "dicom_to_nifti = autorad.utils.preprocessing:dicom_app",
            "nrrd_to_nifti = autorad.utils.preprocessing:nrrd_app",
            "utils = autorad.utils.utils:app",
        ],
    },
)
