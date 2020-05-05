FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY . /app
ENTRYPOINT python detect.py
