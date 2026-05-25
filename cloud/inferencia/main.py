"""Cloud Run SERVICE — sirve la CNN de flores entrenada.

Esto corre en la nube (Cloud Run), NO en Colab. El notebook solo hace POST a
/predict. El modelo se carga desde GCS una vez y queda en memoria.

Endpoints:
  GET  /healthz   -> ¿vivo? ¿modelo cargado?
  GET  /          -> info del modelo y clases
  POST /predict   -> clasifica una imagen

Cargar modelo: por defecto MODEL_GCS (env). Una petición puede pasar otro.

POST /predict  (uno de los dos):
  {"image_gcs": "gs://bucket/demo/flor.jpg"}      # el servicio la lee de GCS
  {"image_b64": "<base64 de un JPG/PNG>"}         # imagen subida
"""
import base64
import io
import json
import os
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from google.cloud import storage
import tensorflow as tf

MODEL_GCS = os.getenv("MODEL_GCS", "")  # gs://bucket/models/flores
IMG = int(os.getenv("IMG_SIZE", "128"))
PORT = int(os.getenv("PORT", "8080"))

gcs = storage.Client()
app = FastAPI(title="cnn-flores-inferencia", version="1")

_lock = threading.Lock()
_state = {"model": None, "model_gcs": None, "classes": [], "load_ms": 0}


def _parse(uri):
    rest = uri.replace("gs://", "")
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def _load(model_gcs: str):
    with _lock:
        if _state["model"] is not None and _state["model_gcs"] == model_gcs:
            return _state
        t0 = time.time()
        # TF lee el SavedModel directo de gs://
        model = tf.saved_model.load(f"{model_gcs.rstrip('/')}/saved_model")
        # clases desde metrics.json
        classes = []
        try:
            bk, prefix = _parse(model_gcs.rstrip("/") + "/metrics.json")
            blob = gcs.bucket(bk).blob(prefix)
            if blob.exists():
                classes = json.loads(blob.download_as_bytes()).get("classes", [])
        except Exception:  # noqa: BLE001
            pass
        _state.update({"model": model, "model_gcs": model_gcs, "classes": classes,
                       "load_ms": int((time.time() - t0) * 1000)})
        return _state


def _leer_imagen(req) -> np.ndarray:
    if req.image_gcs:
        bk, blob = _parse(req.image_gcs)
        data = gcs.bucket(bk).blob(blob).download_as_bytes()
    elif req.image_b64:
        data = base64.b64decode(req.image_b64)
    else:
        raise HTTPException(400, "Pasa image_gcs o image_b64")
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((IMG, IMG))
    return np.expand_dims(np.asarray(img, dtype=np.float32), 0)  # (1,IMG,IMG,3)


class PredictReq(BaseModel):
    image_gcs: str | None = None
    image_b64: str | None = None
    model_gcs: str | None = None


@app.get("/healthz")
def healthz():
    return {"ok": True, "model_loaded": _state["model"] is not None,
            "model_gcs": _state["model_gcs"]}


@app.get("/")
def info():
    return {"model_gcs": _state["model_gcs"], "classes": _state["classes"],
            "img_size": IMG, "load_ms": _state["load_ms"]}


@app.post("/predict")
def predict(req: PredictReq):
    model_gcs = req.model_gcs or MODEL_GCS
    if not model_gcs:
        raise HTTPException(400, "No hay modelo: define MODEL_GCS o pásalo en model_gcs")
    st = _load(model_gcs)
    x = _leer_imagen(req)
    t0 = time.time()
    infer = st["model"].signatures["serving_default"]
    out = infer(tf.constant(x))
    probs = list(out.values())[0].numpy()[0]
    classes = st["classes"] or [str(i) for i in range(len(probs))]
    ranking = sorted(
        ({"clase": classes[i], "prob": round(float(probs[i]) * 100, 1)}
         for i in range(len(probs))),
        key=lambda d: d["prob"], reverse=True)
    return {
        "prediccion": ranking[0]["clase"],
        "confianza": ranking[0]["prob"],
        "ranking": ranking,
        "infer_ms": int((time.time() - t0) * 1000),
        "load_ms": st["load_ms"],
    }


# Pre-carga opcional al arrancar (acelera la primera petición de la demo).
if MODEL_GCS:
    try:
        _load(MODEL_GCS)
    except Exception as e:  # noqa: BLE001
        print("Pre-warm fallo:", e, flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
