"""Utilidades compartidas por los scripts de la demo.

Carga .env, da clientes de GCS y Vision ya configurados, y un par de helpers
para no repetir el mismo boilerplate en cada script de la charla.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Carga demo/.env (si existe) sin sobreescribir variables ya presentes en el shell.
load_dotenv(Path(__file__).with_name(".env"))

PROYECTO = os.getenv("GCP_PROJECT", "")
REGION = os.getenv("GCP_REGION", "europe-southwest1")
BUCKET = os.getenv("BUCKET", "")
DIR_IMAGENES = Path(__file__).with_name("imagenes")


def cliente_storage():
    """Cliente de Google Cloud Storage."""
    from google.cloud import storage
    return storage.Client(project=PROYECTO or None)


def cliente_vision():
    """Cliente de la Vision API (ImageAnnotatorClient)."""
    from google.cloud import vision
    return vision.ImageAnnotatorClient()


def parse_gcs_uri(uri: str):
    """'gs://bucket/ruta/archivo.jpg' -> ('bucket', 'ruta/archivo.jpg')."""
    if not uri.startswith("gs://"):
        raise ValueError(f"No es un URI gs://: {uri}")
    resto = uri[len("gs://"):]
    bucket, _, blob = resto.partition("/")
    return bucket, blob


def imagen_vision_desde(origen: str):
    """Devuelve un vision.Image apuntando a `origen`, que puede ser:
    - un URI gs://bucket/archivo   (Vision lee directo de GCS, sin descargar)
    - una ruta local a un archivo  (se leen los bytes)
    """
    from google.cloud import vision
    if origen.startswith("gs://"):
        return vision.Image(source=vision.ImageSource(image_uri=origen))
    with open(origen, "rb") as f:
        return vision.Image(content=f.read())
