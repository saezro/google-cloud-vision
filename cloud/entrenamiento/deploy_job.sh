#!/usr/bin/env bash
# Despliega y EJECUTA el job de entrenamiento en Cloud Run.
# Uso:  PROJECT=... REGION=europe-southwest1 BUCKET=... bash deploy_job.sh
# (el notebook del taller hace esto mismo en una celda; esto es la versión "a pelo")
set -euo pipefail

: "${PROJECT:?Define PROJECT}"; : "${BUCKET:?Define BUCKET}"
REGION="${REGION:-europe-southwest1}"
JOB="${JOB:-taller-entrenar-flores}"
EPOCHS="${EPOCHS:-8}"
MODEL_DIR="${MODEL_DIR:-models/flores}"
RUNTIME_SA="${RUNTIME_SA:-taller-vision-sa@${PROJECT}.iam.gserviceaccount.com}"

echo "==> Desplegando job '$JOB' (build en la nube con Cloud Build)..."
gcloud run jobs deploy "$JOB" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --service-account "$RUNTIME_SA" \
  --tasks 1 --max-retries 0 \
  --cpu 4 --memory 8Gi --task-timeout 3600 \
  --set-env-vars "BUCKET=$BUCKET,MODEL_DIR=$MODEL_DIR,EPOCHS=$EPOCHS"

echo "==> Ejecutando el job (entrena en la nube y guarda el modelo en gs://$BUCKET/$MODEL_DIR)..."
gcloud run jobs execute "$JOB" --project "$PROJECT" --region "$REGION" --wait

echo "==> Hecho. Modelo en: gs://$BUCKET/$MODEL_DIR/saved_model"
