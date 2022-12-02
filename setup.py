import setuptools
from setuptools import setup


with open("req.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "coverage~=6.2",
    "great-expectations",
    "pytest~=6.2",
    "hypothesis~=6.36",
]

dev_packages = [
    "black~=22.10",
    "flake8~=4.0",
    "isort~=5.10",
    "pre-commit~=2.17",
]

webapp_packages = [
    "streamlit~=1.15",
    "docker~=6.0",
    "jupytext~=1.14",
    "streamlit-extras~=0.2",
]

docs_packages = [
    "mkdocs~=1.3",
    "mkdocs-material~=8.3",
    "mkdocstrings~=0.19",
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
