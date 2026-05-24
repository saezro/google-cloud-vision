#!/usr/bin/env bash
# Despliega y EJECUTA el job de entrenamiento en Cloud Run.
# Uso:  PROJECT=... REGION=europe-southwest1 BUCKET=... bash deploy_job.sh
# (el notebook del taller hace esto mismo en una celda)
set -euo pipefail

: "${PROJECT:?Define PROJECT}"; : "${BUCKET:?Define BUCKET}"
REGION="${REGION:-europe-southwest1}"
JOB="${JOB:-entrenar-flores}"
EPOCHS="${EPOCHS:-8}"

echo "==> Desplegando job '$JOB' (build en la nube con Cloud Build)..."
gcloud run jobs deploy "$JOB" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --tasks 1 --max-retries 0 \
  --cpu 4 --memory 8Gi --task-timeout 3600 \
  --set-env-vars "BUCKET=$BUCKET,EPOCHS=$EPOCHS"

echo "==> Ejecutando el job (entrena en la nube y guarda el modelo en gs://$BUCKET/models/flores)..."
gcloud run jobs execute "$JOB" --project "$PROJECT" --region "$REGION" --wait

echo "==> Hecho. Modelo en: gs://$BUCKET/models/flores/saved_model"
