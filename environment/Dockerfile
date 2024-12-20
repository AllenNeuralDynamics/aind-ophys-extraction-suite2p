FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ENV EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-ophys-extraction-suite2p

ARG DEBIAN_FRONTEND=noninteractive

ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/5619253/tree"
ENV PIPELINE_VERSION="1.0"
ENV VERSION="5.0"

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.1.0 \
    cellpose==3.0 \
    h5py \
    pydantic==2.8.2 \
    sbxreader \
    scikit-image \
    sparse \
    aind-ophys-utils \
    git+https://github.com/j-friedrich/suite2p.git@save_cellpose_output \
    aind-log-utils