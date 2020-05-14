# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with source build for CPU

# taken from https://github.com/microsoft/onnxruntime/blob/master/dockerfiles/Dockerfile.source

# Ubuntu 16.04 Base Image
FROM ubuntu:16.04
MAINTAINER Vinitra Swamy "viswamy@microsoft.com"

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_SERVER_BRANCH=master

RUN apt-get update &&\
    apt-get install -y sudo git bash

WORKDIR /code
ENV PATH /opt/miniconda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:${PATH}

# Prepare onnxruntime repository & build onnxruntime
RUN git clone --single-branch --branch ${ONNXRUNTIME_SERVER_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime cmake-3.14.3-Linux-x86_64

RUN wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx
RUN wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/mask-rcnn/dependencies/coco_classes.txt

COPY requirements.txt /code
RUN pip install -r requirements.txt

# opencv
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev

COPY *.py /code/
ENTRYPOINT python socket_client.py
