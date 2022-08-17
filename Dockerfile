FROM python:3.10-slim

# Install dependencies
COPY setup.py setup.py
COPY setup.cfg setup.cfg
COPY pyproject.toml pyproject.toml
COPY MANIFEST.in MANIFEST.in
COPY requirements.txt requirements.txt
COPY autorad autorad

RUN apt-get update \
    && apt-get install gcc git -y
RUN python -m pip install --upgrade pip && python -m pip install numpy==1.22.4
RUN python -m pip install -e ".[app]"

ENV INPUT_DIR /data
ENV RESULT_DIR /data/results
RUN mkdir -p $INPUT_DIR && mkdir -p $RESULT_DIR

EXPOSE 8501

CMD ["streamlit", "run", "autorad/webapp/app.py"]
