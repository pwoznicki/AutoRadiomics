from setuptools import setup

# Load packages from requirements.txt
with open("requirements.txt", "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    install_requires=required_packages,
    entry_points={
        "console_scripts": [
            "Radiomics = classrad.utils.preprocessing:app",
        ]
    }
)
