<p align="center">
<br>
  <img src="docs/images/logo.png" alt="AutoRadiomics">
</p>

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/pwoznicki/AutoRadiomics/actions/workflows/testing.yml/badge.svg)](https://github.com/pwoznicki/AutoRadiomics/commits/develop)
[![codecov](https://codecov.io/gh/pwoznicki/AutoRadiomics/branch/develop/graph/badge.svg)](https://codecov.io/gh/pwoznicki/AutoRadiomics)

## Framework for simple experimentation with radiomics features

| <p align="center"><a href="https://share.streamlit.io/pwoznicki/autoradiomics/main/webapp/app.py"> Streamlit Share | <p align="center"><a href="https://hub.docker.com/repository/docker/piotrekwoznicki/autorad"> Docker          | <p align="center"><a href="https://pypi.org/project/autorad/"> Python                                          |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/streamlit.png" /></p>  | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/docker.png"/></p> | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/python.png" /></p> |
| <p align="center"><a href="https://share.streamlit.io/pwoznicki/autoradiomics/main/webapp/app.py"> **Demo**        | `docker run -p 8501:8501 -v <your_data_dir>:/data -it piotrekwoznicki/autorad:0.2`                            | `pip install --upgrade autorad`                                                                                |

&nbsp;

### Installation from source

```bash
git clone https://github.com/pwoznicki/AutoRadiomics.git
cd AutoRadiomics
pip install -e .
```

### Getting started

Tutorials can be found in the [examples](./examples/) directory:

- [Binary classification](./examples/example_WORC.ipynb)

Documentation is available at [autoradiomics.readthedocs.io](https://autoradiomics.readthedocs.io/en/latest/).

### Dependencies:

- MONAI
- pyRadiomics
- MLFlow
- Optuna
- scikit-learn
- imbalanced-learn
- XGBoost
- Boruta
- Medpy
- NiBabel
- plotly
- seaborn

### App dependencies:

- Streamlit
- Docker
