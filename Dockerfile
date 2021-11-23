FROM python:3.9-slim

# Install dependencies
COPY setup.py setup.py
COPY setup.cfg setup.cfg
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt
COPY classrad classrad
COPY webapp app

RUN python -m pip install numpy
RUN python -m pip install -e . --no-cache-dir

ENV INPUT_DIR /data
ENV RESULT_DIR /data/results
RUN mkdir -p $INPUT_DIR && mkdir -p $RESULT_DIR

EXPOSE 8501

WORKDIR app
CMD ["streamlit", "run", "app.py"]
