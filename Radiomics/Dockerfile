FROM alpine:3.14

#Installing Python
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y \
    python3.8 \
    python3-pip \
    curl \
    git

#Install dependencies
RUN pip install matplotlib \
                numpy \
                pandas \
                pyradiomics \
                imbalanced-learn==0.4 \
                scikit-learn==0.20.3 \
                tqdm \
                xgboost \
                monai \
                opencv-python \
                nibabel

