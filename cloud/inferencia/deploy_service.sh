#!/usr/bin/env bash
# Despliega el SERVICE de inferencia en Cloud Run.
# Uso:  PROJECT=... REGION=... BUCKET=... bash deploy_service.sh
# (el notebook del taller hace esto en una celda)
set -euo pipefail

: "${PROJECT:?Define PROJECT}"; : "${BUCKET:?Define BUCKET}"
REGION="${REGION:-europe-southwest1}"
SERVICE="${SERVICE:-inferencia-flores}"
MODEL_GCS="${MODEL_GCS:-gs://$BUCKET/models/flores}"

echo "==> Desplegando service '$SERVICE' (build en la nube)..."
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --cpu 2 --memory 4Gi --timeout 120 \
  --allow-unauthenticated \
  --set-env-vars "MODEL_GCS=$MODEL_GCS"

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
        --format='value(status.url)')
echo "==> Service desplegado en: $URL"
echo "    Prueba:  curl $URL/healthz"
