from setuptools import setup

# Load packages from requirements.txt
with open("requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="classrad",
    packages=["classrad"],
    install_requires=required_packages,
)
