"""Genera google-cloud-vision.ipynb — flujo Colab, sin .env.

El notebook se clona a sí mismo + el código + las imágenes con una línea
(`git clone`), te conecta a Google Cloud, te deja elegir el proyecto en un
desplegable y monta todo paso a paso (APIs, IAM, bucket, Vision, entrenar,
inferir). Nada hardcodeado, nada de ficheros de config: solo logueas y eliges.
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

# ============================================================ PORTADA
md("""# Google Cloud Vision — de una foto a una decisión

Dos formas de convertir una imagen en algo útil, una al lado de la otra:

1. **Vision API** — visión por computador ya hecha (etiquetas, objetos, texto). No entrenas nada.
2. **Una CNN tuya** — cuando las clases son tuyas: la entrenas y la sirves como una API, todo en **Cloud Run**.

**Cómo va el cuaderno:** dale al play de arriba a abajo. Cada sección explica *qué* hace y *por qué*.
Casi todo es idempotente —si algo ya existe, no rompe—, así que puedes re-ejecutar sin miedo.""")

md("""### Antes de empezar: Colab **no** es Google Cloud

Es el malentendido número uno, así que vamos claros:

| Dónde corre | Qué hace |
|---|---|
| **Este Colab** (una máquina de Google, *fuera* de tu proyecto) | el **mando**: te autentica, lanza comandos `gcloud`, enseña imágenes y gráficas, llama a los endpoints |
| **Tu proyecto de Google Cloud** | **lo pesado y lo que persiste**: Storage, Vision, **Cloud Run** (entrena y sirve el modelo) |

Consecuencias prácticas: lo que hace este Colab **no** aparece en tu factura de GCP, **no** hay
ninguna VM ni Compute Engine de por medio, y si cierras la pestaña, lo que dejaste en Cloud Run
**sigue vivo**.

**Lo único que damos por hecho:** que ya tienes un **proyecto** de Google Cloud con **billing**
activado (eso se hace una vez, con un par de clics, en la consola web). Todo lo demás lo montamos
aquí, paso a paso.""")

# ============================================================ TRAER EL REPO
md("""## Paso 0 · Traer el código

Una línea baja a esta sesión de Colab el cuaderno, el código de Cloud Run y las imágenes de prueba.""")
code(f'''!git clone -q {REPO_URL} {REPO_DIR} 2>/dev/null || (cd {REPO_DIR} && git pull -q)
%cd {REPO_DIR}
!ls''')

# ============================================================ CONECTAR + PROYECTO
md("""## Paso 1 · Conectar con Google Cloud y elegir proyecto

No hay ningún fichero de configuración: te **conectas con tu cuenta** y eliges el proyecto en un
desplegable. La primera celda abre el diálogo de Google; la segunda lista tus proyectos.""")
code('''# Abre el diálogo de Google para autenticarte (usa la cuenta con tu proyecto)
from google.colab import auth
auth.authenticate_user()
print("Conectado a Google Cloud")''')

md("""Elige tu proyecto en el desplegable que aparece abajo. Cuando lo tengas seleccionado,
**ejecuta la siguiente celda** para fijarlo.""")
code('''# Lista los proyectos de tu cuenta en un desplegable
import ipywidgets as widgets
from IPython.display import display

proyectos = !gcloud projects list --format="value(projectId)" --sort-by=projectId
proyectos = [p for p in proyectos if p.strip()]
_dd = widgets.Dropdown(options=proyectos, description="Proyecto:")
display(_dd)''')

md("""### El resto de la config se deriva sola

A partir del proyecto que elegiste, se calculan el nombre del bucket, la cuenta de servicio, los
nombres del job y del service, etc. No hace falta tocar nada; si quisieras, aquí es donde lo harías.""")
code('''PROJECT    = _dd.value                                  # el que elegiste arriba
REGION     = "europe-southwest1"                        # región de todo (bucket, Cloud Run)
EPOCHS     = 8                                          # épocas de entrenamiento de la CNN
BUCKET     = f"{PROJECT}-imagenes"                      # nombre del bucket (único por proyecto)
MODEL_DIR  = "models/flores"                            # carpeta del modelo dentro del bucket
MODEL_GCS  = f"gs://{BUCKET}/{MODEL_DIR}"               # URI completa del modelo
RUNTIME_SA = f"taller-vision-sa@{PROJECT}.iam.gserviceaccount.com"  # SA con la que corre todo
JOB        = "taller-entrenar-flores"                   # Cloud Run job (entrena)
SERVICE    = "taller-inferencia-flores"                 # Cloud Run service (sirve)

!gcloud config set project {PROJECT} -q
print("Proyecto activo:", PROJECT)
print("Bucket:", BUCKET, "| región:", REGION)''')

# ============================================================ UTILIDADES
md("""## Utilidades

Funciones de apoyo que usamos más abajo (mostrar imágenes, llamar a Vision, pintar gráficas, llamar
al modelo). **No hace falta leerlo** para seguir el taller — ejecútala y a otra cosa.""")
code('''import io, json, time, subprocess
import requests, numpy as np, matplotlib.pyplot as plt
from PIL import Image
!pip -q install google-cloud-vision google-cloud-storage
from google.cloud import storage, vision

_sc  = storage.Client(project=PROJECT)
_vis = vision.ImageAnnotatorClient(client_options={"quota_project_id": PROJECT})
_bucket = lambda: _sc.bucket(BUCKET)

def ver_bucket(prefix="demo/"):
    """Pinta en una fila todas las imágenes que hay en el bucket bajo `prefix`."""
    blobs = [b for b in _sc.list_blobs(BUCKET, prefix=prefix) if not b.name.endswith("/")]
    fig, axes = plt.subplots(1, len(blobs), figsize=(5*len(blobs), 5))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, b in zip(axes, blobs):
        ax.imshow(Image.open(io.BytesIO(b.download_as_bytes()))); ax.axis("off")
        ax.set_title(b.name, fontsize=8)
    plt.show()
    return [f"gs://{BUCKET}/{b.name}" for b in blobs]

def vision_analizar(uri):
    """Pasa una imagen del bucket por Vision: etiquetas, objetos y OCR."""
    img = vision.Image(source=vision.ImageSource(image_uri=uri))
    labels = [(l.description, round(l.score*100, 1)) for l in _vis.label_detection(image=img, max_results=6).label_annotations]
    objs   = [(o.name, round(o.score*100, 1)) for o in _vis.object_localization(image=img).localized_object_annotations]
    t = _vis.text_detection(image=img).text_annotations
    print("ETIQUETAS:", labels)
    print("OBJETOS:  ", objs)
    print("OCR:      ", (t[0].description.strip() if t else "(sin texto)"))
    return {"labels": labels, "objetos": objs}

def decision(uri):
    """Aplica una regla de negocio sencilla sobre la respuesta de Vision: ACEPTAR / REVISAR."""
    r = vision_analizar(uri)
    flores = {"Flower", "Petal", "Plant", "Rose", "Sunflower", "Daisy", "Common sunflower"}
    v = "ACEPTAR" if any(t in flores for t, _ in r["labels"]) else "REVISAR"
    print("-> veredicto:", v)
    return v

def esperar_modelo():
    """Bloquea hasta que el job de entrenamiento haya dejado el modelo en el bucket."""
    print("Esperando al modelo", end="")
    while not _bucket().blob(f"{MODEL_DIR}/metrics.json").exists():
        print(".", end="", flush=True); time.sleep(30)
    print(" listo")

def stats():
    """Lee las métricas que dejó el entrenamiento y pinta accuracy/loss."""
    m = json.loads(_bucket().blob(f"{MODEL_DIR}/metrics.json").download_as_text())
    h, clases = m["history"], m["classes"]
    print(f"val_accuracy: {m['val_accuracy']*100:.1f}%  | clases: {clases}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(h["accuracy"], label="train"); a1.plot(h["val_accuracy"], label="val"); a1.set_title("Accuracy"); a1.legend()
    a2.plot(h["loss"], label="train"); a2.plot(h["val_loss"], label="val"); a2.set_title("Loss"); a2.legend()
    plt.show()
    print("por clase:", m["accuracy_por_clase"])
    return m

def _service_url():
    return subprocess.run(["gcloud", "run", "services", "describe", SERVICE, "--region", REGION,
                           "--format=value(status.url)"], capture_output=True, text=True).stdout.strip()

def clasificar(uri):
    """Llama al modelo servido en Cloud Run (privado: se llama con id-token)."""
    tok = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True).stdout.strip()
    d = requests.post(f"{_service_url()}/predict", json={"image_gcs": uri},
                      headers={"Authorization": f"Bearer {tok}"}, timeout=120).json()
    print(f"{uri}\\n   -> {d['prediccion']}  ({d['confianza']}%)")
    return d

print("Utilidades listas")''')

# ============================================================ 2 · APIs
md("""## Paso 2 · Activar las APIs del proyecto

En Google Cloud cada servicio se activa **por proyecto**. Encendemos los que vamos a usar de una vez:
Storage (el bucket), Vision (la visión gestionada), Cloud Run + Cloud Build + Artifact Registry
(construir y servir contenedores) e IAM (los permisos).""")
code('''!gcloud services enable \\
  storage.googleapis.com vision.googleapis.com \\
  run.googleapis.com cloudbuild.googleapis.com \\
  artifactregistry.googleapis.com iam.googleapis.com -q
print("APIs activadas")''')

# ============================================================ 3 · IAM
md("""## Paso 3 · IAM — quién puede hacer qué

Esta es la parte que más se atasca en la vida real, así que va con calma. La idea de IAM es siempre la
misma: **una identidad** tiene **un rol** sobre **un recurso**.

Montamos dos cosas:

1. Una **service account de runtime**: la "identidad" con la que **corren** nuestros servicios de
   Cloud Run, sin contraseñas ni claves metidas en el código.
2. Los **permisos de despliegue para Cloud Build**: quien construye la imagen y la sube a Cloud Run.

El permiso que todo el mundo olvida es `serviceAccountUser` = **"actuar como"** la otra cuenta. Sin
él, Cloud Build no puede desplegar un servicio que corre *como* la SA de runtime. Es el error nº1.""")
code('''# 3.1 — Crear la service account con la que correrán job y service
!gcloud iam service-accounts create taller-vision-sa \\
  --display-name="Taller Vision runtime" 2>/dev/null || echo "(ya existe)"

# ...y darle acceso al bucket y a consumir APIs
for ROLE in ["roles/storage.admin", "roles/serviceusage.serviceUsageConsumer"]:
    !gcloud projects add-iam-policy-binding {PROJECT} \\
      --member="serviceAccount:{RUNTIME_SA}" --role={ROLE} --condition=None -q > /dev/null
print("Service account de runtime con permisos")''')
code('''# 3.2 — Permisos de despliegue para Cloud Build (NO usamos Compute Engine: todo es Cloud Run)
PNUM = (!gcloud projects describe {PROJECT} --format="value(projectNumber)")[0].strip()
BUILD_SA = f"{PNUM}@cloudbuild.gserviceaccount.com"

for ROLE in ["roles/run.admin", "roles/iam.serviceAccountUser", "roles/artifactregistry.admin",
             "roles/storage.admin", "roles/logging.logWriter", "roles/cloudbuild.builds.builder"]:
    !gcloud projects add-iam-policy-binding {PROJECT} \\
      --member="serviceAccount:{BUILD_SA}" --role={ROLE} --condition=None -q > /dev/null

# "actuar como" la SA de runtime: el permiso que más se olvida
!gcloud iam service-accounts add-iam-policy-binding {RUNTIME_SA} \\
  --member="serviceAccount:{BUILD_SA}" --role="roles/iam.serviceAccountUser" -q > /dev/null
print("Cloud Build listo para desplegar (espera ~30-60s a que el IAM propague)")''')

# ============================================================ 4 · BUCKET
md("""## Paso 4 · Cloud Storage — crear el bucket y subir las imágenes

El **bucket** es el almacén donde viven las imágenes y, más tarde, el modelo entrenado. Lo creamos
ahora (no existía hasta este punto) y subimos las fotos de prueba que vinieron en el repo.""")
code('''# Crear el bucket (idempotente: si ya estaba, no pasa nada)
!gcloud storage buckets create gs://{BUCKET} \\
  --location={REGION} --uniform-bucket-level-access -q || echo "(ya existía)"''')
code('''# Subir las imágenes de prueba del repo (carpeta imagenes/) a gs://BUCKET/demo/
import glob
for ruta in sorted(glob.glob("imagenes/*.jpg")):
    !gcloud storage cp "{ruta}" gs://{BUCKET}/demo/ -q
IMG = f"gs://{BUCKET}/demo/flor.jpg"   # imagen por defecto para los ejemplos de abajo
print("Subidas:", sorted(glob.glob("imagenes/*.jpg")))''')
md("Y las vemos *desde el bucket* (se descargan de GCS, no son las locales):")
code('''ver_bucket()''')

# ============================================================ 5 · ENTRENAR
md("""## Paso 5 · Entrenar tu propia CNN desde cero en Cloud Run (job)

Aquí está el corazón del taller: **entrenamos un modelo desde cero** con nuestras propias clases
(5 tipos de flor). Lo hacemos con un **Cloud Run job** —un contenedor que arranca, entrena y muere—
construido desde `cloud/entrenamiento/`. Corre **como la SA de runtime** y recibe toda la config por
variables de entorno. Lo lanzamos en segundo plano (`--async`, ~10-15 min) y seguimos mientras tanto.

> En la sesión, el modelo ya está entrenado en el bucket de antes, así que el siguiente paso no
> espera: verás las métricas al momento mientras este job corre por detrás.""")
code('''%cd cloud/entrenamiento
!gcloud run jobs deploy {JOB} --source . --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --cpu 4 --memory 8Gi --task-timeout 3600 --max-retries 0 \\
  --set-env-vars BUCKET={BUCKET},MODEL_DIR={MODEL_DIR},EPOCHS={EPOCHS} -q
%cd ../..
!gcloud run jobs execute {JOB} --region {REGION} --async
print("Entrenando en la nube. Llama a esperar_modelo() cuando quieras el resultado.")''')

# ============================================================ 6 · STATS
md("""## Paso 6 · Ver cómo ha aprendido el modelo

El entrenamiento deja en el bucket un `metrics.json` con todo el historial. La siguiente celda
**espera** a que esté (si ya estaba, sigue al momento) y pinta las curvas de accuracy y loss.""")
code('''esperar_modelo()
m = stats()''')

# ============================================================ 7 · INFERIR
md("""## Paso 7 · Servir el modelo en Cloud Run e inferir

Ya tienes el modelo entrenado; ahora lo pones a trabajar. Desplegamos el otro tipo de Cloud Run: un
**service** (siempre disponible, escala a 0 cuando no se usa) que carga el modelo desde el bucket y
responde a peticiones. Se construye desde `cloud/inferencia/`.

> La organización bloquea el acceso público, así que el service queda **privado** y lo llamamos
> autenticados con un id-token (lo hace `clasificar()` por dentro). Es, de hecho, la práctica
> recomendada aunque pudieras abrirlo.""")
code('''%cd cloud/inferencia
!gcloud run deploy {SERVICE} --source . --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --cpu 2 --memory 4Gi --timeout 120 --min-instances 0 \\
  --set-env-vars MODEL_GCS={MODEL_GCS} -q
%cd ../..
print("Service desplegado en:", _service_url())''')
md("Y clasificamos las tres flores de prueba contra **nuestro** modelo, el que acabamos de entrenar:")
code('''clasificar(IMG)
clasificar(f"gs://{BUCKET}/demo/rosa.jpg")
clasificar(f"gs://{BUCKET}/demo/margarita.jpg")''')

# ============================================================ 8 · VISION (contraste, al final)
md("""## Paso 8 · Y además: la Vision API (esto, pero ya hecho)

Acabas de **entrenar y servir** un modelo a mano. Conviene saber que, para clases **genéricas**,
Google ya tiene un modelo entrenado y servido que usas como servicio: la **Vision API**. Le mandas
una imagen y te devuelve etiquetas, objetos y texto (OCR) ya reconocidos — **tú no entrenas ni
despliegas nada**.

Lo probamos sobre la misma foto y le ponemos encima una **regla de negocio** trivial (¿hay una flor?
→ ACEPTAR, si no → REVISAR), para ver el contraste con lo que construimos arriba:""")
code('''decision(IMG)''')
md("""**¿Cuándo cada uno?**

| | Vision API | Tu CNN en Cloud Run |
|---|---|---|
| Entrenamiento | ninguno | lo entrenas tú |
| Clases | generales (las de Google) | **las tuyas** |
| Puesta en marcha | inmediata | construir + desplegar |
| Control | el que da la API | total |

Regla práctica: si tus clases están entre las que Google ya reconoce, tira de **Vision API**; si son
específicas de tu negocio (defectos, piezas, documentos tuyos…), no hay atajo: **modelo propio**,
justo lo que montaste en los pasos 5-7.""")

# ============================================================ 9 · CIERRE
md("""## Paso 9 · Costes y limpieza

**Costes:** con `--min-instances 0` el service no cuesta nada en reposo; el job solo cuesta mientras
entrena; la Vision API regala 1.000 usos al mes. Todo es **CPU, sin GPU**.

**Limpieza** (se lleva por delante todo lo creado):

```
!gcloud projects delete {PROJECT}
```""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "name": "google-cloud-vision.ipynb"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "google-cloud-vision.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("escrito google-cloud-vision.ipynb con", len(cells), "celdas")
