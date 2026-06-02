#!/usr/bin/env bash
# Despliega el SERVICE de inferencia en Cloud Run.
# Uso:  PROJECT=... REGION=... BUCKET=... bash deploy_service.sh
# (el notebook del taller hace esto en una celda; esto es la versión "a pelo")
set -euo pipefail

: "${PROJECT:?Define PROJECT}"; : "${BUCKET:?Define BUCKET}"
REGION="${REGION:-europe-southwest1}"
SERVICE="${SERVICE:-taller-inferencia-flores}"
MODEL_GCS="${MODEL_GCS:-gs://$BUCKET/models/flores}"
RUNTIME_SA="${RUNTIME_SA:-taller-vision-sa@${PROJECT}.iam.gserviceaccount.com}"

# El service queda PRIVADO (sin --allow-unauthenticated): muchas orgs bloquean
# 'allUsers' por org policy. Se llama autenticado con un id-token (ver abajo).
echo "==> Desplegando service '$SERVICE' (build en la nube)..."
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --service-account "$RUNTIME_SA" \
  --cpu 2 --memory 4Gi --timeout 180 --min-instances 0 \
  --set-env-vars "MODEL_GCS=$MODEL_GCS"

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
        --format='value(status.url)')
echo "==> Service desplegado en: $URL"
echo "    Prueba (privado, con id-token):"
echo "      curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $URL/"
