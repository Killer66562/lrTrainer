FROM python:3.11-slim

COPY src/datasets /mnt/datasets

WORKDIR /opt
RUN mkdir lr

COPY src/train.py lr
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "lr/train.py" ]