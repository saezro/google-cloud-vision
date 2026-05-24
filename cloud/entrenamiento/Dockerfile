# Cloud Run JOB de entrenamiento. No expone puerto: corre y termina.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py .

# Cloud Run job ejecuta este comando hasta completar.
ENTRYPOINT ["python", "train.py"]
