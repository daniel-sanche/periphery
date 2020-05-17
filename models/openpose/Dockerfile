FROM python:3.7 as builder

FROM mcr.microsoft.com/azureml/onnxruntime:v1.2.0

COPY requirements.txt /code
RUN pip install -r requirements.txt

COPY human-pose-estimation.onnx /code

COPY *.py /code/
ENTRYPOINT python socket_client.py
