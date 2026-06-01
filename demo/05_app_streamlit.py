"""05 — App visual del "resultado de negocio" (opcional, para el wow).

Sube una foto en el navegador y ve, al instante: etiquetas, cajas de objetos
dibujadas, OCR y el VEREDICTO de negocio. Ideal para audiencia mixta porque no
hay que mirar la terminal.

    streamlit run 05_app_streamlit.py

Usa las mismas funciones que la demo de terminal (03 y 04), así que lo que ven
en la app es exactamente lo que corre por código.
"""
import io
import tempfile

import streamlit as st
from PIL import Image, ImageDraw
from importlib import import_module

from comun import cliente_vision, imagen_vision_desde

v = import_module("03_vision_basico")
neg = import_module("04_pipeline_negocio")

st.set_page_config(page_title="Imagen → resultado de negocio", page_icon="", layout="wide")
st.title(" De una imagen a un resultado de negocio")
st.caption("Google Cloud Storage + Vision API — demo de la charla")

subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

if subida:
    data = subida.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Guardamos a un temporal para reutilizar el pipeline tal cual.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp, format="JPEG")
        ruta = tmp.name

    client = cliente_vision()
    image = imagen_vision_desde(ruta)
    objs = v.objetos(client, image)

    # Dibuja las cajas de objetos sobre la imagen
    anotada = img.copy()
    draw = ImageDraw.Draw(anotada)
    W, H = anotada.size
    for o in objs:
        xs = [p[0] * W for p in o["caja"]]
        ys = [p[1] * H for p in o["caja"]]
        draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=(255, 90, 0), width=3)
        draw.text((min(xs) + 4, min(ys) + 4), f"{o['nombre']} {o['score']:.0f}%", fill=(255, 90, 0))

    col1, col2 = st.columns(2)
    with col1:
        st.image(anotada, caption="Objetos detectados (Vision API)", use_container_width=True)
        st.subheader("Etiquetas")
        for desc, score in v.etiquetas(client, image):
            st.write(f"- **{desc}** · {score:.0f}%")
        ocr = v.texto(client, image)
        if ocr:
            st.subheader("Texto (OCR)")
            st.code(ocr)

    with col2:
        reg = neg.evaluar(ruta)
        color = {"OK": "green", "REVISAR": "orange", "RECHAZADA": "red"}.get(reg["veredicto"], "gray")
        st.markdown(f"### Veredicto de negocio: :{color}[{reg['veredicto']}]")
        if reg["motivos"]:
            st.warning("Motivos: " + "; ".join(reg["motivos"]))
        st.metric("Objetos fiables", reg["n_objetos"])
        if reg["series_detectadas"]:
            st.write("**Series detectadas:**", ", ".join(reg["series_detectadas"]))
        st.subheader("Registro estructurado")
        st.json(reg)
        st.caption("Esto es lo que tu sistema mandaría a una BD / n8n / Sheet.")
