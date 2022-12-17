# Installation

There are a few ways to install and use AutoRadiomics, depending on your needs.

If you want to use the code base, you can install the package from PyPI:

```bash
pip install autorad
```

You can also install it from source:

```bash
git clone https://github.com/pwoznicki/AutoRadiomics.git
cd AutoRadiomics
pip install -e .
```

If you want to only use the web application, we suggest that you use the docker image that is always in sync with the latest version of the code:

```bash
docker run -p 8501:8501 -v <your_data_dir>:/data -it pwoznicki/autorad:latest
```

&nbsp;

# Tutorials

Tutorials can be found in the [examples](https://github.com/pwoznicki/AutoRadiomics/tree/main/examples) directory:

- [Binary classification](https://github.com/pwoznicki/AutoRadiomics/tree/main/examples/example_WORC.ipynb)

&nbsp;

# Web application

To use the application, make sure you have its dependencies installed:

```bash
pip install -e ".[app]"
```

The application can be started from the root directory with:

```bash
streamlit run autorad/webapp/app.py
```

By default it willl run at http://localhost:8501/.
<br/><br/>
