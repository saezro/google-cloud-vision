"""03 — Consumir la Vision API: las 4 capacidades clave.

Objetivos #3 y #4: "Consumir la API de Vision" y "Analizar imágenes
automáticamente". Una función por capacidad, para que en la charla se vea
nítido qué da cada una. Acepta un gs:// (Vision lee directo de GCS) o una ruta
local.

    python 03_vision_basico.py gs://mi-bucket/demo/foto.jpg
    python 03_vision_basico.py imagenes/foto.jpg
"""
import sys

from comun import cliente_vision, imagen_vision_desde


def etiquetas(client, image, n=8):
    """¿Qué hay en la imagen? (clasificación general, modelo pre-entrenado)."""
    resp = client.label_detection(image=image, max_results=n)
    return [(l.description, round(l.score * 100, 1)) for l in resp.label_annotations]


def objetos(client, image):
    """Detección + localización de objetos (esto es lo que hace un YOLO, pero
    sin entrenar nada: devuelve clase + caja normalizada 0..1)."""
    resp = client.object_localization(image=image)
    out = []
    for o in resp.localized_object_annotations:
        verts = [(round(v.x, 3), round(v.y, 3)) for v in o.bounding_poly.normalized_vertices]
        out.append({"nombre": o.name, "score": round(o.score * 100, 1), "caja": verts})
    return out


def texto(client, image):
    """OCR: extrae texto de la imagen (placas, números de serie, etiquetas)."""
    resp = client.text_detection(image=image)
    if not resp.text_annotations:
        return ""
    return resp.text_annotations[0].description.strip()


def seguridad(client, image):
    """SafeSearch: ¿la imagen es válida o hay que descartarla/moderarla?"""
    resp = client.safe_search_detection(image=image)
    s = resp.safe_search_annotation
    nivel = lambda v: v.name  # UNKNOWN/VERY_UNLIKELY/.../VERY_LIKELY
    return {"adult": nivel(s.adult), "violence": nivel(s.violence),
            "racy": nivel(s.racy), "medical": nivel(s.medical)}


def analizar(origen: str):
    client = cliente_vision()
    image = imagen_vision_desde(origen)

    print(f"\n=== Analizando: {origen} ===\n")

    print("ETIQUETAS (qué hay en la foto):")
    for desc, score in etiquetas(client, image):
        print(f"  - {desc:25s} {score:5.1f}%")

    print("\nOBJETOS LOCALIZADOS (detección tipo YOLO, gestionada):")
    objs = objetos(client, image)
    if objs:
        for o in objs:
            print(f"  - {o['nombre']:20s} {o['score']:5.1f}%  caja={o['caja'][0]}..{o['caja'][2]}")
    else:
        print("  (ninguno)")

    print("\nTEXTO (OCR):")
    t = texto(client, image)
    print("  " + (t.replace("\n", "\n  ") if t else "(sin texto)"))

    print("\nSAFE SEARCH (¿imagen válida?):")
    for k, v in seguridad(client, image).items():
        print(f"  - {k:10s} {v}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python 03_vision_basico.py <gs://... | ruta-local>")
    analizar(sys.argv[1])
