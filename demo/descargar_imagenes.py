"""Prepara imágenes de ejemplo en demo/imagenes/.

- Intenta descargar 3 fotos de dominio público (Wikimedia) con objetos, texto y
  una escena de calle: buenas para enseñar etiquetas / objetos / OCR.
- SIEMPRE genera además una "placa" sintética con un número de serie (AT-12345)
  para que el demo de OCR y de negocio funcione aunque no haya red.

    python descargar_imagenes.py
"""
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

DIR = Path(__file__).with_name("imagenes")
DIR.mkdir(exist_ok=True)

# (filename, url). URLs de Wikimedia Commons (dominio público / CC).
FUENTES = [
    ("calle.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Bisesero_Genocide_Memorial_Site.jpg/640px-Bisesero_Genocide_Memorial_Site.jpg"),
    ("producto.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Canned_food_and_drinks_in_a_pantry.jpg/640px-Canned_food_and_drinks_in_a_pantry.jpg"),
    ("senal.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Stop_sign_MUTCD.svg/480px-Stop_sign_MUTCD.svg.png"),
]


def descargar():
    req_headers = {"User-Agent": "charla-vision-demo/1.0 (educativo)"}
    for nombre, url in FUENTES:
        destino = DIR / nombre
        if destino.exists():
            print(f"· ya existe {nombre}")
            continue
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                destino.write_bytes(r.read())
            print(f" descargada {nombre}")
        except Exception as e:  # noqa: BLE001
            print(f" no pude descargar {nombre} ({e}). Sigo.")


def placa_sintetica():
    """Imagen con un número de serie legible — OCR garantizado para el demo."""
    destino = DIR / "placa_activo.png"
    img = Image.new("RGB", (640, 240), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 620, 220], outline=(40, 40, 40), width=4)
    d.text((50, 60), "TALLER VISION · ACTIVO DEMO", fill=(20, 20, 20))
    d.text((50, 110), "SERIE: DEMO-48291", fill=(0, 0, 0))
    d.text((50, 150), "MODELO: CAM-2024-X", fill=(60, 60, 60))
    img.save(destino)
    print(f" generada {destino.name} (serie DEMO-48291)")


if __name__ == "__main__":
    descargar()
    placa_sintetica()
    print(f"\nImágenes en {DIR}/ . Siguiente:  python 02_subir_imagenes.py")
