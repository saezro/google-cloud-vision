# Cloud Run SERVICE de inferencia. Expone HTTP en $PORT (Cloud Run lo inyecta).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# Cloud Run manda el puerto en $PORT. uvicorn lo respeta vía la app.
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
