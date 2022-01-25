import setuptools
from setuptools import setup

with open("requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "coverage==6.2",
    "great-expectations==0.14.2",
    "pytest==6.2.5",
    "hypothesis==6.36.0",
]

dev_packages = [
    "black==21.12b0",
    "flake8==4.0.1",
    "isort==5.10.1",
    "pre-commit==2.17.0",
]

webapp_packages = ["streamlit==1.2.0", "docker==5.0.3"]

setup(
    name="classrad",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=required_packages,
    extras_require={"dev": test_packages + dev_packages + webapp_packages},
)
