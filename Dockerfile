FROM python:3.11-slim

WORKDIR /opt
RUN mkdir lr
RUN mkdir lr/models

COPY src/train.py lr
COPY src/datasets lr/datasets

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/lr

ENTRYPOINT [ "python", "train.py" ]