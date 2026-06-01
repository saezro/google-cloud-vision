"""02 — Subir imágenes al bucket.

Objetivo #2 de la sesión: "Subir imágenes". Sube todo lo que haya en
demo/imagenes/ y te imprime las URIs gs:// (que luego Vision lee directo, sin
descargar la imagen a tu máquina).

    python 02_subir_imagenes.py
"""
from comun import cliente_storage, BUCKET, DIR_IMAGENES

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def subir_todo(prefijo: str = "demo"):
    if not BUCKET:
        raise SystemExit("Define BUCKET en demo/.env.")
    archivos = sorted(p for p in DIR_IMAGENES.glob("*") if p.suffix.lower() in EXTS)
    if not archivos:
        raise SystemExit(
            f"No hay imágenes en {DIR_IMAGENES}. Corre antes:  python descargar_imagenes.py"
        )
    bucket = cliente_storage().bucket(BUCKET)
    uris = []
    for ruta in archivos:
        destino = f"{prefijo}/{ruta.name}"
        blob = bucket.blob(destino)
        blob.upload_from_filename(str(ruta))
        uri = f"gs://{BUCKET}/{destino}"
        uris.append(uri)
        print(f" subida  {ruta.name:30s} -> {uri}")
    print(f"\n{len(uris)} imágenes en el bucket. Copia una URI para el siguiente paso:")
    print(f"    python 03_vision_basico.py {uris[0]}")
    return uris


if __name__ == "__main__":
    subir_todo()
