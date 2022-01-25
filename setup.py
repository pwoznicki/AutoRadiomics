import setuptools
from setuptools import setup

with open("requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="classrad",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=required_packages,
)
