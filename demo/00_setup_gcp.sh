#!/usr/bin/env bash
# ============================================================================
#  Setup GCP para la charla — PROYECTO APARTE (NO tocar tu-proyecto-gcp)
# ============================================================================
#  Ejecútalo paso a paso (no de un tirón) la primera vez. Cada bloque te dice
#  qué hace. Coste esperado: ~0 € (Vision API: 1000 unidades/mes gratis;
#  Storage: unos pocos MB de imágenes de demo).
#
#  REQUISITO: estar logueado con una cuenta CON billing y permiso de crear
#  proyectos. El SA tu-cuenta@example.com NO sirve para esto.
#      gcloud auth login            # tu-usuario@example.com  o  tu-usuario@example.com
# ============================================================================
set -euo pipefail

# ---- Edita estas 3 variables --------------------------------------------------
PROJECT_ID="charla-vision-demo"          # debe ser único global; añade sufijo si peta
REGION="europe-southwest1"               # Madrid
BUCKET="charla-vision-demo-imagenes"     # debe ser único global
# Billing account: lístalas con `gcloud billing accounts list` y pega el ID aquí
BILLING_ACCOUNT=""                       # p.ej. 0X0X0X-0X0X0X-0X0X0X
# ------------------------------------------------------------------------------

echo "==> 1. Crear el proyecto (aparte de tu-proyecto-gcp)"
gcloud projects create "$PROJECT_ID" --name="Charla Vision Demo"

echo "==> 2. Vincular billing (necesario para activar APIs)"
[ -n "$BILLING_ACCOUNT" ] || { echo "Rellena BILLING_ACCOUNT (gcloud billing accounts list)"; exit 1; }
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"

echo "==> 3. Fijar el proyecto activo para esta sesión"
gcloud config set project "$PROJECT_ID"

echo "==> 4. Activar las APIs que usa la charla"
gcloud services enable vision.googleapis.com storage.googleapis.com

echo "==> 5. Crear el bucket de imágenes"
gcloud storage buckets create "gs://$BUCKET" \
  --project="$PROJECT_ID" --location="$REGION" --uniform-bucket-level-access

echo "==> 6. Service Account para la demo (credencial que usará Python)"
SA="charla-demo"
SA_EMAIL="${SA}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "$SA" --display-name="Charla Vision demo"
# Storage: admin solo sobre ESTE bucket (mínimo privilegio)
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.admin"
# Vision no necesita rol de proyecto especial para llamar a la API una vez activada;
# basta con que la cuenta pueda autenticarse. Si tu org lo exige, añade:
#   gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#     --member="serviceAccount:${SA_EMAIL}" --role="roles/serviceusage.serviceUsageConsumer"

echo "==> 7. Descargar la key del SA a la ubicación canónica (chmod 600)"
KEY_DIR="$HOME/.config/gcp/sa-keys/$PROJECT_ID"
mkdir -p "$KEY_DIR"; chmod 700 "$HOME/.config/gcp" "$HOME/.config/gcp/sa-keys" "$KEY_DIR"
gcloud iam service-accounts keys create "$KEY_DIR/$SA.json" --iam-account="$SA_EMAIL"
chmod 600 "$KEY_DIR/$SA.json"

cat <<EOF

============================================================
 LISTO. Ahora copia demo/.env.example a demo/.env y pon:
   GCP_PROJECT=$PROJECT_ID
   GCP_REGION=$REGION
   BUCKET=$BUCKET
   GOOGLE_APPLICATION_CREDENTIALS=$KEY_DIR/$SA.json
============================================================

 Alternativa SIN service account (más rápido para ensayar):
   gcloud auth application-default login
   ...y deja GOOGLE_APPLICATION_CREDENTIALS vacío en .env

 LIMPIEZA tras la charla (borra TODO el proyecto y su gasto):
   gcloud projects delete $PROJECT_ID
EOF
