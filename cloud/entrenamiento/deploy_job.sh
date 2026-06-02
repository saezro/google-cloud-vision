#!/usr/bin/env bash
# Despliega y EJECUTA el job de entrenamiento (CON GPU) en Cloud Run.
# Uso:  PROJECT=... REGION=europe-west4 BUCKET=... bash deploy_job.sh
# (el notebook del taller hace esto mismo en una celda; esto es la versión "a pelo")
#
# OJO: requiere una región con GPU NVIDIA L4 (p.ej. europe-west4, europe-west1,
# us-central1) y cuota de GPU en el proyecto.
set -euo pipefail

: "${PROJECT:?Define PROJECT}"; : "${BUCKET:?Define BUCKET}"
REGION="${REGION:-europe-west4}"
JOB="${JOB:-taller-entrenar-flores}"
EPOCHS="${EPOCHS:-40}"
IMG_SIZE="${IMG_SIZE:-180}"
MODEL_DIR="${MODEL_DIR:-models/flores102}"
RUNTIME_SA="${RUNTIME_SA:-taller-vision-sa@${PROJECT}.iam.gserviceaccount.com}"

echo "==> Desplegando job '$JOB' con GPU L4 (build en la nube con Cloud Build)..."
gcloud run jobs deploy "$JOB" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --service-account "$RUNTIME_SA" \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --tasks 1 --max-retries 0 \
  --cpu 4 --memory 16Gi --task-timeout 3600 \
  --set-env-vars "BUCKET=$BUCKET,MODEL_DIR=$MODEL_DIR,EPOCHS=$EPOCHS,IMG_SIZE=$IMG_SIZE"

# El job NO tiene coste en reposo: arranca, entrena, libera la GPU y muere.
echo "==> Ejecutando el job (entrena en GPU y guarda el modelo en gs://$BUCKET/$MODEL_DIR)..."
gcloud run jobs execute "$JOB" --project "$PROJECT" --region "$REGION" --wait

echo "==> Hecho. Modelo en: gs://$BUCKET/$MODEL_DIR/saved_model"
