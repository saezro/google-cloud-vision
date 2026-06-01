"""04 — De la imagen al RESULTADO DE NEGOCIO.

Objetivo #5: "Integrar el proceso desde Python". Aquí está el corazón de la
charla: una imagen no es un dato hasta que una REGLA DE NEGOCIO la convierte en
una decisión. Este script:

  imagen  ->  Vision API  ->  extrae señales  ->  aplica regla  ->  REGISTRO + veredicto

El registro se imprime como JSON y se acumula en registros.csv (lo que tu
sistema real mandaría a una base de datos, un n8n, un Sheet...).

Es el puente conceptual a un pipeline con modelo propio: allí la "regla" la
pone una CNN que entrenas tú; aquí es Vision + un poco de lógica.

    python 04_pipeline_negocio.py gs://mi-bucket/demo/foto.jpg
"""
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from comun import cliente_vision, imagen_vision_desde
from importlib import import_module

# Reutilizamos las funciones del script 03 (módulo "03_vision_basico").
v = import_module("03_vision_basico")

CSV = Path(__file__).with_name("registros.csv")

# --- Parámetros de la regla de negocio (ajústalos a tu caso) -------------------
OBJETOS_DE_INTERES = {"Packaged goods", "Box", "Product", "Person", "Car",
                      "Solar panel", "Hardware", "Tin can", "Bottle"}
PATRON_SERIE = re.compile(r"\b[A-Z]{2,4}[-_ ]?\d{3,8}\b")  # p.ej. AT-12345
UMBRAL_CONFIANZA = 60.0  # % mínimo para contar un objeto


def evaluar(origen: str) -> dict:
    client = cliente_vision()
    image = imagen_vision_desde(origen)

    # 1) Señales crudas de Vision
    labels = v.etiquetas(client, image, n=5)
    objs = v.objetos(client, image)
    ocr = v.texto(client, image)
    safe = v.seguridad(client, image)

    # 2) Derivar datos de negocio
    objetos_fiables = [o for o in objs if o["score"] >= UMBRAL_CONFIANZA]
    de_interes = [o for o in objetos_fiables if o["nombre"] in OBJETOS_DE_INTERES]
    series = sorted(set(PATRON_SERIE.findall(ocr.upper())))
    imagen_invalida = safe["adult"] in {"LIKELY", "VERY_LIKELY"} or \
        safe["violence"] in {"LIKELY", "VERY_LIKELY"}

    # 3) REGLA DE NEGOCIO -> veredicto + motivos
    motivos = []
    if imagen_invalida:
        veredicto = "RECHAZADA"
        motivos.append("SafeSearch marca contenido no apto")
    else:
        if not objetos_fiables:
            motivos.append("no se detecta ningún objeto con confianza suficiente")
        if not series:
            motivos.append("no se ha podido leer un número de serie")
        veredicto = "OK" if not motivos else "REVISAR"

    registro = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "imagen": origen,
        "veredicto": veredicto,
        "motivos": motivos,
        "n_objetos": len(objetos_fiables),
        "objetos_interes": [o["nombre"] for o in de_interes],
        "series_detectadas": series,
        "etiqueta_top": labels[0][0] if labels else None,
        "safe_search": safe,
    }
    return registro


def guardar_csv(reg: dict):
    nuevo = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["ts", "imagen", "veredicto", "n_objetos",
                        "objetos_interes", "series", "etiqueta_top", "motivos"])
        w.writerow([reg["ts"], reg["imagen"], reg["veredicto"], reg["n_objetos"],
                    "|".join(reg["objetos_interes"]), "|".join(reg["series_detectadas"]),
                    reg["etiqueta_top"], "; ".join(reg["motivos"])])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python 04_pipeline_negocio.py <gs://... | ruta-local>")
    reg = evaluar(sys.argv[1])
    print(json.dumps(reg, indent=2, ensure_ascii=False))
    guardar_csv(reg)
    icono = {"OK": "", "REVISAR": "", "RECHAZADA": ""}.get(reg["veredicto"], "")
    print(f"\n{icono} Veredicto: {reg['veredicto']}   (registro añadido a {CSV.name})")
