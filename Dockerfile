FROM alpine:3.14

#Installing Python
RUN apk update
RUN apk add --no-cache python3 \
    py3-pip \
    git

#clone the repo with scripts
WORKDIR /workspace
RUN git clone https://github.com/piotrekwoznicki/Radiomics.git

#Change branch and install dependencies
WORKDIR /workspace/Radiomics
RUN git checkout radiomics_maps
RUN pip install -r requirements.txt

