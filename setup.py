import setuptools
from setuptools import setup

with open("requirements.txt") as file:
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
    "jupytext~=1.14",
]

docs_packages = [
    "mkdocs==1.4.2",
    "mkdocs-material==8.5.10",
    "mkdocstrings[python]==0.19.0",
]

setup(
    name="autorad",
    packages=setuptools.find_packages(),
    package_data={
        "autorad": [
            "config/pyradiomics_feature_names.json",
            "webapp/paths_example.csv",
            "webapp/templates/segmentation/pretrained_models.json",
            "webapp/templates/segmentation/nnunet_code.py.jinja",
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
        "console_scripts": [],
    },
)
