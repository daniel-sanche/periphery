FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . /app
ENTRYPOINT python socket_client.py
