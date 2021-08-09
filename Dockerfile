FROM python:3.7-slim

#Installing Python
RUN apt-get update
RUN apt-get install -y git

#clone the repo with scripts
WORKDIR /workspace
RUN git clone https://github.com/piotrekwoznicki/Radiomics.git

#Change branch and install dependencies
WORKDIR /workspace/Radiomics
RUN git checkout radiomics_maps
RUN pip install --no-cache -r requirements.txt

