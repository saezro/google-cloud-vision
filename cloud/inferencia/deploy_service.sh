#!/usr/bin/env bash
# Despliega el SERVICE de inferencia en Cloud Run.
# Uso:  PROJECT=... REGION=... BUCKET=... bash deploy_service.sh
# (el notebook del taller hace esto en una celda; esto es la versión "a pelo")
set -euo pipefail

: "${PROJECT:?Define PROJECT}"; : "${BUCKET:?Define BUCKET}"
REGION="${REGION:-europe-west4}"
SERVICE="${SERVICE:-taller-inferencia-flores}"
MODEL_GCS="${MODEL_GCS:-gs://$BUCKET/models/flores102}"
RUNTIME_SA="${RUNTIME_SA:-taller-vision-sa@${PROJECT}.iam.gserviceaccount.com}"

# Inferencia EN GPU (--gpu 1 --gpu-type nvidia-l4).
# El service queda PRIVADO (sin --allow-unauthenticated): muchas orgs bloquean
# 'allUsers' por org policy. Se llama autenticado con un id-token (ver abajo).
# --min-instances 0 -> CLAVE con GPU: se apaga solo y deja de cobrar la GPU en reposo.
echo "==> Desplegando service '$SERVICE' con GPU (build en la nube)..."
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --service-account "$RUNTIME_SA" \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 4 --memory 16Gi --timeout 180 --min-instances 0 --max-instances 1 \
  --set-env-vars "MODEL_GCS=$MODEL_GCS"

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
        --format='value(status.url)')
echo "==> Service desplegado en: $URL"
echo "    Prueba (privado, con id-token):"
echo "      curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $URL/"
