FROM python:3.8.2-slim

ADD requirements.txt ./
RUN pip install -r requirements.txt

ADD client.py ./

ENTRYPOINT python client.py
