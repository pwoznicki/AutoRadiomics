<p align="center">
<br>
  <img src="docs/images/logo.png" alt="AutoRadiomics">
</p>

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/pwoznicki/AutoRadiomics/actions/workflows/testing.yml/badge.svg)](https://github.com/pwoznicki/AutoRadiomics/commits/main)
[![codecov](https://codecov.io/gh/pwoznicki/AutoRadiomics/branch/develop/graph/badge.svg)](https://codecov.io/gh/pwoznicki/AutoRadiomics)

## Framework for simple experimentation with radiomics features

| <p align="center"><a href="https://pwoznicki-autoradiomics-autoradwebappapp-feature-desktop-4lpmpi.streamlitapp.com"> Streamlit Share | <p align="center"><a href="https://hub.docker.com/repository/docker/pwoznicki/autorad"> Docker          | <p align="center"><a href="https://pypi.org/project/autorad/"> Python                                          |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/streamlit.png" /></p>  | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/docker.png"/></p> | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/python.png" /></p> |
| <p align="center"><a href="https://pwoznicki-autoradiomics-autoradwebappapp-feature-desktop-4lpmpi.streamlitapp.com"> **Demo**        | `docker run -p 8501:8501 -v <your_data_dir>:/data -it pwoznicki/autorad:0.2.4`                            | `pip install -U autorad`                                                                                |

&nbsp;


## Download desktop app (experimental)
| <p align="center"><a href="https://drive.google.com/uc?export=download&id=1fZyBeMvFUZXn7ND_FgeQRV3W68Dn6zZb"> Windows 10 | <p align="center"><a href="https://drive.google.com/uc?export=download&id=1N3JLv2h00Pp8XfwWXbBWvr7OnQ2h9pNu"> MacOS 11 (x64) | <p align="center"><a href="https://drive.google.com/uc?export=download&id=1SDG7J5ucwd4Nkq-5fAeArLKvHTcD045M"> Ubuntu 20.04                                          |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/windows.png" /></p>  | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/macos.png"/></p> | <p align="center"><img src="https://github.com/pwoznicki/AutoRadiomics/raw/main/docs/images/ubuntu.png" /></p> |


## Installation from source

```bash
git clone https://github.com/pwoznicki/AutoRadiomics.git
cd AutoRadiomics
pip install -e .
```

## Getting started

Tutorials can be found in the [examples](./examples/) directory:

- [Binary classification](./examples/example_WORC.ipynb)

Documentation is available at [autoradiomics.readthedocs.io](https://autoradiomics.readthedocs.io/en/latest/).

## Web application

The application can be started from the root directory with:

```
streamlit run autorad/webapp/app.py
```

By default it willl run at http://localhost:8501/.
<br/><br/>

For more information about AutoRadiomics, please read [our paper](https://www.frontiersin.org/articles/10.3389/fradi.2022.919133/full):
```
  AutoRadiomics: A Framework for Reproducible Radiomics Research;
  P Woznicki, F Laqua, T Bley, B Baeßler;
  Frontiers in Radiology, 22
```
Please cite it if you're using the framework for your research.
