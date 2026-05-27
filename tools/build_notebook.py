"""Genera taller.ipynb — flujo GIT CLONE (para Colab).

El notebook se clona a sí mismo + el código + las imágenes con una línea
(`git clone`), carga la config de `.env`, y despliega en Cloud Run desde las
carpetas del repo. Nada incrustado, nada hardcodeado.
"""
import json
from pathlib import Path

ROOT = Path("/home/saez/Code/charla-vision-gcs")
REPO_URL = "https://github.com/saezro/google-cloud-vision.git"
REPO_DIR = "gcv"

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.rstrip("\n").splitlines(keepends=True)})

md("""# Google Cloud Vision — de una foto a una decisión

Vision API + una CNN tuya, entrenada y servida en **Cloud Run**. Colab solo manda; el cómputo va en
la nube. Dale al play de arriba a abajo (casi todo es idempotente: si algo ya existe, no rompe).""")

md("""### Colab NO es Google Cloud — es solo el **mando**

| Dónde corre | Qué hace |
|---|---|
| **Este Colab** (fuera de GCP) | autentica, lanza `gcloud`, enseña imágenes/gráficas, llama a los endpoints |
| **Google Cloud** (el proyecto GCP) | **lo pesado**: Storage, Vision, **Cloud Run** (entrena y sirve) |

No cuesta en el billing · **no hay VM / Compute Engine**. Al cerrar Colab, lo de Cloud Run sigue vivo.""")

# ---- CLONE ----
md("## Traer el repo (una línea: código + imágenes + config)")
code(f'''!git clone -q {REPO_URL} {REPO_DIR} 2>/dev/null || (cd {REPO_DIR} && git pull -q)
%cd {REPO_DIR}
!ls''')

# ---- 0 CONFIG ----
md("""## 0 · Configuración

Asumimos que el proyecto GCP y el billing **ya están creados** (eso se hace por la **UI de la consola**).
Si hay un `.env`, se usa; si no, nos **conectamos a Google Cloud y se listan los proyectos** para
elegir uno en un desplegable, sin escribir nada a mano.""")
code('''# Conectar con Google Cloud (abre el diálogo de Google)
from google.colab import auth
auth.authenticate_user()
print("Conectado a Google Cloud")''')
code('''# Elegir proyecto: desde .env si existe; si no, lista los proyectos de la cuenta
import os
from pathlib import Path
!pip -q install python-dotenv
from dotenv import load_dotenv

if Path(".env").exists():
    load_dotenv(".env", override=True)
    PROJECT = os.environ["PROJECT"]
    _dd = None
    print("Proyecto (desde .env):", PROJECT)
else:
    import ipywidgets as widgets
    from IPython.display import display
    proyectos = !gcloud projects list --format="value(projectId)"
    proyectos = [p for p in proyectos if p.strip()]
    _dd = widgets.Dropdown(options=proyectos, description="Proyecto:")
    display(_dd)
    print("No hay .env -> elige el proyecto en el desplegable y ejecuta la siguiente celda.")''')
code('''# Config final (se deriva todo del PROJECT) + fijar el proyecto activo
PROJECT    = os.environ.get("PROJECT") or _dd.value
REGION     = os.environ.get("REGION", "europe-southwest1")
EPOCHS     = int(os.environ.get("EPOCHS", "8"))
BUCKET     = os.environ.get("BUCKET", f"{PROJECT}-imagenes")
MODEL_DIR  = os.environ.get("MODEL_DIR", "models/flores")
MODEL_GCS  = f"gs://{BUCKET}/{MODEL_DIR}"
RUNTIME_SA = os.environ.get("RUNTIME_SA", f"taller-vision-sa@{PROJECT}.iam.gserviceaccount.com")
JOB        = os.environ.get("JOB", "taller-entrenar-flores")
SERVICE    = os.environ.get("SERVICE", "taller-inferencia-flores")
!gcloud config set project {PROJECT} -q
print("Proyecto activo:", PROJECT, "| bucket:", BUCKET)''')

# ---- UTILIDADES ----
md("## Utilidades — *no hace falta leerlo* (funciones que usamos abajo)")
code('''import io, json, time, subprocess
import requests, numpy as np, matplotlib.pyplot as plt
from PIL import Image
!pip -q install google-cloud-vision google-cloud-storage
from google.cloud import storage, vision
_sc  = storage.Client(project=PROJECT)
_vis = vision.ImageAnnotatorClient(client_options={"quota_project_id": PROJECT})
_bucket = lambda: _sc.bucket(BUCKET)

def ver_bucket(prefix="demo/"):
    blobs = [b for b in _sc.list_blobs(BUCKET, prefix=prefix) if not b.name.endswith("/")]
    fig, axes = plt.subplots(1, len(blobs), figsize=(5*len(blobs),5))
    axes = axes if hasattr(axes,"__len__") else [axes]
    for ax,b in zip(axes,blobs):
        ax.imshow(Image.open(io.BytesIO(b.download_as_bytes()))); ax.axis("off"); ax.set_title(b.name, fontsize=8)
    plt.show(); return [f"gs://{BUCKET}/{b.name}" for b in blobs]

def vision_analizar(uri):
    img = vision.Image(source=vision.ImageSource(image_uri=uri))
    labels = [(l.description, round(l.score*100,1)) for l in _vis.label_detection(image=img, max_results=6).label_annotations]
    objs   = [(o.name, round(o.score*100,1)) for o in _vis.object_localization(image=img).localized_object_annotations]
    t = _vis.text_detection(image=img).text_annotations
    print("ETIQUETAS:", labels, "\\nOBJETOS:", objs, "\\nOCR:", (t[0].description.strip() if t else "(sin texto)"))
    return {"labels": labels, "objetos": objs}

def decision(uri):
    r = vision_analizar(uri)
    flores = {"Flower","Petal","Plant","Rose","Sunflower","Daisy","Common sunflower"}
    v = "ACEPTAR" if any(t in flores for t,_ in r["labels"]) else "REVISAR"
    print("-> veredicto:", v); return v

def esperar_modelo():
    print("Esperando al modelo", end="")
    while not _bucket().blob(f"{MODEL_DIR}/metrics.json").exists():
        print(".", end="", flush=True); time.sleep(30)
    print(" ")

def stats():
    m = json.loads(_bucket().blob(f"{MODEL_DIR}/metrics.json").download_as_text())
    h, clases = m["history"], m["classes"]
    print(f"val_accuracy: {m['val_accuracy']*100:.1f}%  | clases: {clases}")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    a1.plot(h["accuracy"],label="train"); a1.plot(h["val_accuracy"],label="val"); a1.set_title("Accuracy"); a1.legend()
    a2.plot(h["loss"],label="train"); a2.plot(h["val_loss"],label="val"); a2.set_title("Loss"); a2.legend()
    plt.show(); print("por clase:", m["accuracy_por_clase"]); return m

def _service_url():
    return subprocess.run(["gcloud","run","services","describe",SERVICE,"--region",REGION,
                           "--format=value(status.url)"], capture_output=True, text=True).stdout.strip()

def clasificar(uri):
    tok = subprocess.run(["gcloud","auth","print-identity-token"], capture_output=True, text=True).stdout.strip()
    d = requests.post(f"{_service_url()}/predict", json={"image_gcs": uri},
                      headers={"Authorization": f"Bearer {tok}"}, timeout=120).json()
    print(f" {uri}\\n   -> {d['prediccion']}  ({d['confianza']}%)"); return d

print("utilidades listas ")''')

# ---- 1 APIs ----
md("""## 1 · Activar las APIs
Cada servicio se activa por proyecto: Storage, Vision, Cloud Run, Cloud Build, Artifact Registry, IAM.""")
code('''!gcloud services enable \\
  storage.googleapis.com vision.googleapis.com \\
  run.googleapis.com cloudbuild.googleapis.com \\
  artifactregistry.googleapis.com iam.googleapis.com -q
print("APIs activadas ")''')

# ---- 2 IAM ----
md("""## 2 · IAM — quién puede hacer qué  

**Una IDENTIDAD tiene un ROL sobre un RECURSO.** Creamos una **service account** con la que
**corren** los servicios (sin contraseñas en el código) y damos a **Cloud Build** los roles para
construir y desplegar. Clave: `serviceAccountUser` = "actuar como" la SA (el error nº1 de Cloud Run).""")
code('''# 2.1 — SA de runtime
!gcloud iam service-accounts create taller-vision-sa --display-name="Taller Vision runtime" 2>/dev/null || echo "(ya existe)"
for ROLE in ["roles/storage.admin", "roles/serviceusage.serviceUsageConsumer"]:
    !gcloud projects add-iam-policy-binding {PROJECT} --member="serviceAccount:{RUNTIME_SA}" --role={ROLE} --condition=None -q > /dev/null
print("Runtime SA con permisos ")''')
code('''# 2.2 — Permisos de despliegue a Cloud Build (NO usamos Compute Engine: todo Cloud Run)
PNUM = (!gcloud projects describe {PROJECT} --format="value(projectNumber)")[0].strip()
BUILD_SA = f"{PNUM}@cloudbuild.gserviceaccount.com"
for ROLE in ["roles/run.admin","roles/iam.serviceAccountUser","roles/artifactregistry.admin",
             "roles/storage.admin","roles/logging.logWriter","roles/cloudbuild.builds.builder"]:
    !gcloud projects add-iam-policy-binding {PROJECT} --member="serviceAccount:{BUILD_SA}" --role={ROLE} --condition=None -q > /dev/null
!gcloud iam service-accounts add-iam-policy-binding {RUNTIME_SA} --member="serviceAccount:{BUILD_SA}" --role="roles/iam.serviceAccountUser" -q > /dev/null
print("Cloud Build con permisos de despliegue  (espera ~30-60s a que propague)")''')

# ---- 3 BUCKET ----
md("## 3 · Cloud Storage — bucket + imágenes (las del repo)")
code('''!gcloud storage buckets create gs://{BUCKET} --location={REGION} --uniform-bucket-level-access -q || echo "(ya existía)"''')
code('''# Subir las imágenes de prueba del repo (carpeta imagenes/)
import glob
for ruta in sorted(glob.glob("imagenes/*.jpg")):
    !gcloud storage cp "{ruta}" gs://{BUCKET}/demo/ -q
IMG = f"gs://{BUCKET}/demo/flor.jpg"
print("Subidas:", sorted(glob.glob("imagenes/*.jpg")))''')
code('''ver_bucket()''')

# ---- 4 VISION ----
md("## 4 · Vision API — visión gestionada → decisión de negocio")
code('''decision(IMG)''')

# ---- 5 ENTRENAR ----
md("""## 5 · Entrenar la CNN en Cloud Run (job)

Desplegamos el **job** desde `cloud/entrenamiento/` (corre como la SA, config por env vars) y lo
ejecutamos en segundo plano (~10-15 min). Mientras, seguimos.""")
code('''%cd cloud/entrenamiento
!gcloud run jobs deploy {JOB} --source . --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --cpu 4 --memory 8Gi --task-timeout 3600 --max-retries 0 \\
  --set-env-vars BUCKET={BUCKET},MODEL_DIR={MODEL_DIR},EPOCHS={EPOCHS} -q
%cd ../..
!gcloud run jobs execute {JOB} --region {REGION} --async
print(" Entrenando en la nube. Usa esperar_modelo() para el resultado.")''')

# ---- 6 STATS ----
md("## 6 · Estadísticas del modelo (cuando acabe)")
code('''esperar_modelo()
m = stats()''')

# ---- 7 INFERIR ----
md("""## 7 · Servir la CNN en Cloud Run e inferir

Desplegamos el **service** desde `cloud/inferencia/`. La org bloquea el acceso público, así que es
**privado** y `clasificar()` lo llama autenticado con un id-token.""")
code('''%cd cloud/inferencia
!gcloud run deploy {SERVICE} --source . --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --cpu 2 --memory 4Gi --timeout 120 --min-instances 0 \\
  --set-env-vars MODEL_GCS={MODEL_GCS} -q
%cd ../..
print("Service:", _service_url())''')
code('''clasificar(IMG)
clasificar(f"gs://{BUCKET}/demo/rosa.jpg")
clasificar(f"gs://{BUCKET}/demo/margarita.jpg")''')

# ---- 8 CIERRE ----
md("""## 8 · Comparativa, costes y limpieza

- **Vision API**: 0 entrenamiento, clases generales, inmediato.
- **Tu CNN**: TUS clases, la entrenaste tú en Cloud Run, control total.

Coste: con `--min-instances 0` el service no cuesta en reposo; el job solo al entrenar; **sin GPU**.

Limpieza total: `!gcloud projects delete {PROJECT}`""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "name": "google-cloud-vision.ipynb"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "google-cloud-vision.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("escrito google-cloud-vision.ipynb con", len(cells), "celdas")
