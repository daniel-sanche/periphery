FROM python:3.7 as builder

FROM mcr.microsoft.com/azureml/onnxruntime:v1.2.0

COPY requirements.txt /code
RUN pip install -r requirements.txt

# opencv
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev

COPY human-pose-estimation.onnx /code

COPY *.py /code/
ENTRYPOINT python socket_client.py
