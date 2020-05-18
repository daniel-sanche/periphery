FROM mcr.microsoft.com/azureml/onnxruntime:v1.2.0

RUN wget https://github.com/onnx/models/raw/master/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx

COPY fix_model.py .
RUN python fix_model.py

COPY requirements.txt /code
RUN pip install -r requirements.txt

# opencv
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev

RUN wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
COPY *.py /code/
ENTRYPOINT python socket_client.py
