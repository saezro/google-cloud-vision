"""01 — Crear un bucket de GCS desde Python.

En la charla esto demuestra el objetivo #1: "Crear un bucket en Google Cloud
Storage". Normalmente el bucket ya lo creó 00_setup_gcp.sh; este script enseña
que también se hace en 5 líneas de Python (idempotente: si ya existe, no falla).

    python 01_crear_bucket.py
"""
from comun import cliente_storage, BUCKET, REGION


def crear_bucket(nombre: str = BUCKET, region: str = REGION):
    client = cliente_storage()
    existente = client.lookup_bucket(nombre)   # None si no existe
    if existente:
        print(f" El bucket gs://{nombre} ya existe (location={existente.location}).")
        return existente
    bucket = client.create_bucket(nombre, location=region)
    print(f" Bucket creado: gs://{bucket.name} en {bucket.location}.")
    return bucket


if __name__ == "__main__":
    if not BUCKET:
        raise SystemExit("Define BUCKET en demo/.env (copia .env.example).")
    crear_bucket()
