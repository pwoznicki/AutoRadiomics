FROM alpine:3.14

#Installing Python
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y \
    python3.7 \
    python3-pip \
    curl \
    git

#clone the repo with scripts
WORKDIR /workspace
RUN git clone https://github.com/piotrekwoznicki/Radiomics.git

#Change branch and install dependencies
WORKDIR /workspace/Radiomics
RUN git checkout radiomics_maps
RUN pip install -r requirements.txt

