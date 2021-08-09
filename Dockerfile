FROM python:3.7-alpine

#Installing Python
RUN apk update
RUN apk add --no-cache git

#clone the repo with scripts
WORKDIR /workspace
RUN git clone https://github.com/piotrekwoznicki/Radiomics.git

#Change branch and install dependencies
WORKDIR /workspace/Radiomics
RUN git checkout radiomics_maps
RUN pip3 install -r requirements.txt

