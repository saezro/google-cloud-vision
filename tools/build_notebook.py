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
md("""# Entrenar y servir tu propio modelo en Google Cloud

De un dataset de imágenes a un modelo en producción, **todo en tu nube**:

1. **Entrenas** una CNN desde cero con tus propias clases, en un **job de Cloud Run**.
2. La **sirves** como una API en un **service de Cloud Run** y haces inferencia sobre fotos nuevas.

El modelo es **tuyo** y corre en **tu** infraestructura: tú lo entrenas, tú lo despliegas, tú lo
llamas. Nada pre-hecho ni externo.

**Cómo va el cuaderno:** dale al play de arriba a abajo. Cada sección explica *qué* hace y *por qué*.
Casi todo es idempotente —si algo ya existe, no rompe—, así que puedes re-ejecutar sin miedo.""")

md("""### Antes de empezar: Colab **no** es Google Cloud

Es el malentendido número uno, así que vamos claros:

| Dónde corre | Qué hace |
|---|---|
| **Este Colab** (una máquina de Google, *fuera* de tu proyecto) | el **mando**: te autentica, lanza comandos `gcloud`, enseña imágenes y gráficas, llama a los endpoints |
| **Tu proyecto de Google Cloud** | **lo pesado y lo que persiste**: Storage, **Cloud Run** (entrena y sirve el modelo) |

Consecuencias prácticas: lo que hace este Colab **no** aparece en tu factura de GCP, **no** hay
ninguna VM ni Compute Engine de por medio, y si cierras la pestaña, lo que dejaste en Cloud Run
**sigue vivo**.

**Lo único que damos por hecho:** que ya tienes un **proyecto** de Google Cloud con **billing**
activado (eso se hace una vez, con un par de clics, en la consola web). Todo lo demás lo montamos
aquí, paso a paso.""")

# ============================================================ CONCEPTOS
md("""## Conceptos · las piezas de Google Cloud que vamos a usar

Antes de tocar nada, el mapa. Todo en GCP cuelga de un **proyecto** (la unidad de facturación y de
permisos). Dentro vamos a usar cinco piezas:

**1 · Cloud Storage — el almacén.** Guardas ficheros (imágenes, el modelo entrenado…) en **buckets**.
Cada objeto tiene una URI tipo `gs://mi-bucket/carpeta/archivo.jpg`. Es el disco compartido al que
todo lo demás se conecta.

**2 · IAM — quién puede hacer qué.** La regla de oro: **una identidad** tiene **un rol** sobre **un
recurso**. Una identidad puede ser una persona o una **service account** (una "cuenta de robot" con la
que corren los programas, sin contraseñas). En el taller creamos una service account para que nuestros
contenedores tengan permiso de leer/escribir en el bucket.

**3 · Cloud Build + Artifact Registry — construir el contenedor.** Tú entregas tu código y un
`Dockerfile`; Cloud Build lo empaqueta en una **imagen de contenedor** y la guarda en Artifact
Registry. Esa imagen es lo que Cloud Run ejecuta.

**4 · Cloud Run — donde corre tu código.** Ejecuta contenedores sin que gestiones servidores. Y aquí
está la distinción clave del taller: Cloud Run tiene **dos formas** de correr algo.""")

md("""### Cloud Run JOB vs SERVICE — la diferencia que hay que tener clara

| | **Job** (tarea) | **Service** (servicio) |
|---|---|---|
| Qué es | un contenedor que **arranca, hace una tarea y muere** | un contenedor que **se queda escuchando** peticiones HTTP |
| Vive | mientras dura el trabajo, luego se apaga | siempre disponible (escala a 0 cuando nadie lo usa) |
| Cómo se invoca | lo **ejecutas** (`jobs execute`) | le **mandas peticiones** (HTTP a su URL) |
| Para qué, aquí | **entrenar** el modelo (empieza, entrena, termina) | **servir** el modelo (responde a cada inferencia) |
| Analogía | un horno: lo enciendes, cocina, se apaga | un restaurante: abierto, atiende cuando llega gente |

Es el eje del taller: **entrenas con un JOB** (Paso 5) y **sirves con un SERVICE** (Pasos 7-8). Mismo
Cloud Run, dos modos para dos necesidades distintas.

Con el mapa claro, vamos a montarlo.""")

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
code('''PROJECT       = _dd.value                               # el que elegiste arriba
REGION        = "europe-southwest1"                     # región de todo (bucket, Cloud Run)
EPOCHS        = 8                                        # épocas de entrenamiento de la CNN
BUCKET        = f"{PROJECT}-imagenes"                    # nombre del bucket (único por proyecto)

MODEL_DIR     = "models/flores"                          # tu CNN entrenada (carpeta en el bucket)
MODEL_GCS     = f"gs://{BUCKET}/{MODEL_DIR}"             # URI de tu modelo
PRETRAIN_DIR  = "models/imagenet"                        # modelo pre-entrenado (lo descargamos)
PRETRAIN_GCS  = f"gs://{BUCKET}/{PRETRAIN_DIR}"          # URI del modelo pre-entrenado

RUNTIME_SA = f"taller-vision-sa@{PROJECT}.iam.gserviceaccount.com"  # SA con la que corre todo
JOB        = "taller-entrenar-flores"                   # Cloud Run job (entrena)
SERVICE    = "taller-inferencia-flores"                 # Cloud Run service (sirve cualquier modelo)

!gcloud config set project {PROJECT} -q
print("Proyecto activo:", PROJECT)
print("Bucket:", BUCKET, "| región:", REGION)''')

# ============================================================ UTILIDADES
md("""## Utilidades

Funciones de apoyo que usamos más abajo (mostrar imágenes del bucket, esperar al entrenamiento,
pintar las gráficas, llamar al modelo servido). **No hace falta leerlo** para seguir el taller —
ejecútala y a otra cosa.""")
code('''import io, json, time, subprocess
import requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
!pip -q install google-cloud-storage
from google.cloud import storage

_sc  = storage.Client(project=PROJECT)
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

def clasificar(uri, modelo=None):
    """Llama al modelo servido en Cloud Run. `modelo` elige cuál (por defecto, el del service).

    El service es agnóstico al modelo: si le pasas `model_gcs`, sirve ese. Así el MISMO
    endpoint clasifica con tu CNN o con el modelo pre-entrenado, según a cuál apuntes."""
    tok = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True).stdout.strip()
    payload = {"image_gcs": uri}
    if modelo:
        payload["model_gcs"] = modelo
    d = requests.post(f"{_service_url()}/predict", json=payload,
                      headers={"Authorization": f"Bearer {tok}"}, timeout=180).json()
    print(f"{uri}\\n   -> {d['prediccion']}  ({d['confianza']}%)   top: {[r['clase'] for r in d['ranking']]}")
    return d

def registro_modelos():
    """Inventario de modelos = leer la 'ficha' (metrics.json) de cada carpeta en models/.

    No hay base de datos aparte: el propio bucket es el registro. Cada modelo guarda su
    metrics.json al entrenarse/subirse, y listándolos tienes tu catálogo con dónde está
    cada uno y cómo llamarlo."""
    filas = []
    for b in _sc.list_blobs(BUCKET, prefix="models/"):
        if not b.name.endswith("/metrics.json"):
            continue
        meta = json.loads(b.download_as_text())
        carpeta = b.name.rsplit("/", 1)[0]            # models/<nombre>
        filas.append({
            "modelo": carpeta.split("/", 1)[1],
            "descripcion": meta.get("modelo", "CNN entrenada (flores)"),
            "n_clases": len(meta.get("classes", [])),
            "img_size": meta.get("img_size"),
            "val_accuracy": meta.get("val_accuracy"),     # None si es pre-entrenado
            "actualizado": b.updated.strftime("%Y-%m-%d %H:%M") if b.updated else None,
            "ruta_gcs": f"gs://{BUCKET}/{carpeta}",        # <- dónde está; esto es lo que pasas a clasificar()
        })
    return pd.DataFrame(filas).sort_values("modelo").reset_index(drop=True)

def dibujar_cnn_3d(model):
    """Dibuja las capas conv/pool como volúmenes 3D apilados (alto×ancho = mapa espacial,
    grosor = nº de canales). El diagrama clásico de una CNN: el mapa encoge, la profundidad crece."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    colores = {"Entrada": "#9AA7B0", "Conv2D": "#4C72B0", "MaxPooling2D": "#55A868",
               "GlobalAveragePooling2D": "#8172B3", "Dense": "#DD8452"}
    # caja de entrada + solo las capas que cambian de forma (saltamos normalizado y aumento de datos)
    e = tuple(model.layers[0].output.shape)  # forma de entrada (salida del Rescaling)
    cajas = [("Entrada", e[1], e[2], e[3], f"{e[1]}×{e[2]}×{e[3]}")]
    for L in model.layers:
        n = L.__class__.__name__
        if n not in ("Conv2D", "MaxPooling2D", "Dense", "GlobalAveragePooling2D"):
            continue
        s = tuple(L.output.shape)
        if len(s) == 4:
            _, h, w, c = s; cajas.append((n, h, w, c, f"{h}×{w}×{c}"))
        else:
            _, u = s; cajas.append((n, max(u/8, 2), max(u/8, 2), u, f"{u}"))
    fig = plt.figure(figsize=(15, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    x = 0.0
    for i, (nombre, h, w, c, etiq) in enumerate(cajas):
        dy, dz = w / 8.0, h / 8.0
        dx = max(c / 16.0, 0.3)
        col = colores.get(nombre, "#BBBBBB")
        x0, y0, z0 = x, -dy / 2, -dz / 2
        v = np.array([[x0,y0,z0],[x0+dx,y0,z0],[x0+dx,y0+dy,z0],[x0,y0+dy,z0],
                      [x0,y0,z0+dz],[x0+dx,y0,z0+dz],[x0+dx,y0+dy,z0+dz],[x0,y0+dy,z0+dz]])
        caras = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
                 [v[2],v[3],v[7],v[6]],[v[1],v[2],v[6],v[5]],[v[0],v[3],v[7],v[4]]]
        ax.add_collection3d(Poly3DCollection(caras, facecolor=col, edgecolor="white",
                                             linewidths=0.4, alpha=0.92))
        z_lab = dz/2 + (1.2 if i % 2 == 0 else 3.6)   # alternar altura para no solapar
        ax.text(x0+dx/2, 0, z_lab, etiq, ha="center", va="bottom", fontsize=8)
        x += dx + 3.0
    ax.set_xlim(0, x); ax.set_ylim(-9, 9); ax.set_zlim(-9, 11)
    ax.set_axis_off(); ax.view_init(elev=18, azim=-68)
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, c in colores.items()],
              loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(); plt.show()

def dibujar_arquitectura():
    """Mapa de TODO el taller: qué hace cada parte y DÓNDE corre (Colab vs tu proyecto GCP)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 7.8)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    AZUL, VERDE, NARANJA, GRIS, MORADO = "#4C72B0", "#55A868", "#DD8452", "#7F7F7F", "#8172B3"

    def caja(x, y, w, h, color, titulo, lineas, fc=None, tfs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                     linewidth=2, edgecolor=color, facecolor=fc or "white", alpha=0.95))
        ax.text(x+w/2, y+h-4, titulo, ha="center", va="top", fontsize=tfs, fontweight="bold", color=color)
        for i, ln in enumerate(lineas):
            ax.text(x+3, y+h-11-i*4.4, ln, ha="left", va="top", fontsize=8.3, color="#222")

    def flecha(x1, y1, x2, y2, txt="", color="#444", rad=0.0, lpos=None, astyle="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=astyle, mutation_scale=15,
                     lw=1.7, color=color, connectionstyle=f"arc3,rad={rad}"))
        if txt:
            lx, ly = lpos or ((x1+x2)/2, (y1+y2)/2+1.5)
            ax.text(lx, ly, txt, ha="center", va="bottom", fontsize=7.6, color=color, style="italic")

    ax.add_patch(FancyBboxPatch((34, 5), 64, 90, boxstyle="round,pad=0.5,rounding_size=3",
                 linewidth=2, edgecolor="#999", facecolor="#F4F6F8", alpha=0.6, linestyle="--"))
    ax.text(66, 92, "TU PROYECTO DE GOOGLE CLOUD  ·  aquí corre lo pesado", ha="center",
            va="center", fontsize=11, fontweight="bold", color="#666")

    caja(1.5, 14, 28, 64, GRIS, "COLAB · el mando",
         ["0 · git clone (trae el código)", "1 · login + elegir proyecto",
          "2 · activar APIs", "3 · crear IAM", "4 · subir imágenes",
          "5 · lanzar el job", "7 · desplegar el service", "8 · pedir inferencias",
          "9 · inventario de modelos"], fc="#F0F0F0")
    caja(40, 58, 54, 22, AZUL, "Cloud Storage · bucket",
         ["demo/                 imágenes de prueba", "models/flores         tu CNN entrenada",
          "models/imagenet   MobileNet descargado"], fc="#EAF0F7")
    caja(40, 31, 25, 18, VERDE, "Cloud Run JOB", ["entrena la CNN", "(arranca y muere)"], fc="#EAF3EE")
    caja(69, 31, 25, 18, NARANJA, "Cloud Run SERVICE", ["sirve cualquier", "modelo (HTTP)"], fc="#FBEFE6")
    caja(40, 9, 54, 13, MORADO, "IAM + Cloud Build",
         ["SA de runtime · construye y despliega los contenedores"], fc="#F0EDF6")

    flecha(29.5, 60, 40, 67, "sube imágenes", AZUL, 0.12, lpos=(34, 66))
    flecha(29.5, 50, 40, 40, "deploy + execute", VERDE, 0.05, lpos=(34.5, 47))
    flecha(53, 49, 58, 58, "guarda modelo", VERDE, -0.2, lpos=(49, 53))
    flecha(76, 58, 81, 49, "carga modelo", NARANJA, -0.2, lpos=(85, 53))
    flecha(69, 34, 29.5, 41, "pide inferencia  ·  recibe predicción", NARANJA, -0.3,
           lpos=(49, 24.5), astyle="<|-|>")

    ax.text(66, 2.3, "Colab no es Google Cloud: solo da órdenes. Si cierras Colab, lo de Cloud Run sigue vivo.",
            ha="center", va="center", fontsize=9, style="italic", color="#666")
    plt.tight_layout(); plt.show()

print("Utilidades listas")''')


# ============================================================ 2 · APIs
md("""## Paso 2 · Activar las APIs del proyecto

En Google Cloud cada servicio se activa **por proyecto**. Encendemos los que vamos a usar de una vez:
Storage (el bucket), Cloud Run + Cloud Build + Artifact Registry (construir y servir contenedores) e
IAM (los permisos).""")
code('''!gcloud services enable \\
  storage.googleapis.com \\
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
él, Cloud Build no puede desplegar un servicio que corre *como* la SA de runtime. Es el error nº1.

> **Quién construye, en realidad.** `gcloud run deploy --source` empaqueta tu código con Cloud Build.
> Desde 2024 ese build corre, por defecto, como la **service account por defecto de Compute Engine**
> (`NUMERO-compute@developer...`) — no como la antigua `@cloudbuild`. Como no sabemos la antigüedad de
> tu proyecto, damos los permisos a **las dos** para que no falle. Es el error nº2 (y muy puñetero,
> porque el mensaje habla de "storage.objects.get" y despista).""")
code('''# 3.1 — Crear la service account con la que correrán job y service
!gcloud iam service-accounts create taller-vision-sa \\
  --display-name="Taller Vision runtime" 2>/dev/null || echo "(ya existe)"

# ...y darle acceso al bucket y a consumir APIs
for ROLE in ["roles/storage.admin", "roles/serviceusage.serviceUsageConsumer"]:
    !gcloud projects add-iam-policy-binding {PROJECT} \\
      --member="serviceAccount:{RUNTIME_SA}" --role={ROLE} --condition=None -q > /dev/null
print("Service account de runtime con permisos")''')
code('''# 3.2 — Permisos de despliegue para quien construye (las DOS posibles SA de build)
PNUM = (!gcloud projects describe {PROJECT} --format="value(projectNumber)")[0].strip()
BUILD_SAS = [f"{PNUM}-compute@developer.gserviceaccount.com",  # builder por defecto (2024+)
             f"{PNUM}@cloudbuild.gserviceaccount.com"]         # builder antiguo

ROLES = ["roles/run.admin", "roles/iam.serviceAccountUser", "roles/artifactregistry.admin",
         "roles/storage.admin", "roles/logging.logWriter", "roles/cloudbuild.builds.builder"]
for SA in BUILD_SAS:
    for ROLE in ROLES:
        !gcloud projects add-iam-policy-binding {PROJECT} \\
          --member="serviceAccount:{SA}" --role={ROLE} --condition=None -q > /dev/null
    # "actuar como" la SA de runtime: el permiso que más se olvida
    !gcloud iam service-accounts add-iam-policy-binding {RUNTIME_SA} \\
      --member="serviceAccount:{SA}" --role="roles/iam.serviceAccountUser" -q > /dev/null
print("Permisos de build concedidos (espera ~30-60s a que el IAM propague)")''')

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
(5 tipos de flor). Antes de lanzarlo, veamos **qué** vamos a entrenar.""")

md("""### La arquitectura de la CNN

Esta es la red que usamos — pequeña pero suficiente para el caso (con ~60 mil parámetros llega a un
`val_accuracy ≈ 0.62` sobre 5 clases). La idea de una CNN: las **capas convolucionales** detectan
patrones cada vez más complejos (primero bordes, luego texturas, luego formas) y el **clasificador**
final convierte eso en una probabilidad por clase.

| Bloque | Capas | Qué hace |
|---|---|---|
| Entrada y normalizado | `Input` + `Rescaling` | imagen 128×128×3, píxeles a rango 0-1 |
| Aumento de datos | `RandomFlip`, `RandomRotation` | variaciones al vuelo (solo al entrenar) para generalizar mejor |
| Extracción ×3 | `Conv2D` + `MaxPooling2D` | aprenden patrones y reducen el tamaño (128→63→30→14) |
| Resumen | `GlobalAveragePooling2D` | cada mapa de características → un número (64 en total) |
| Clasificador | `Dense(64)` + `Dropout` | combina las características; el dropout evita sobreajuste |
| Salida | `Dense(softmax)` | una probabilidad por clase (5 flores) |

Veamos primero **cómo se construye**, capa a capa. Es **la misma función** que usa el job (la
importamos del repo y enseñamos su código tal cual, sin copiar nada):""")
code('''# Importamos la arquitectura REAL del job y mostramos su código de construcción
import sys, inspect
sys.path.insert(0, "cloud/entrenamiento")
from train import construir_modelo

print(inspect.getsource(construir_modelo))''')
md("La instanciamos (5 clases, 128×128) y miramos su resumen — capas, formas y nº de parámetros:")
code('''modelo_demo = construir_modelo(n_clases=5, img=128)
modelo_demo.summary()''')
md("""Y en **3D**, para verlo de un vistazo: cada bloque es el volumen de datos que sale de esa capa.
Se ve el embudo típico de una CNN — el **mapa espacial encoge** (128→63→30→14) mientras la
**profundidad de canales crece** (3→32→64), hasta que el clasificador lo reduce a 5 probabilidades.""")
code('''dibujar_cnn_3d(modelo_demo)''')

md("""### Lanzar el entrenamiento

Ahora sí: desplegamos el **Cloud Run job** desde `cloud/entrenamiento/` —un contenedor que arranca,
entrena y muere—. Corre **como la SA de runtime** y recibe la config por variables de entorno. Lo
lanzamos en segundo plano (`--async`, ~2-3 min) y seguimos mientras tanto.

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
md("""## Paso 7 · Servir tu modelo en Cloud Run e inferir

Ya tienes el modelo entrenado; ahora lo pones a trabajar. Desplegamos el otro tipo de Cloud Run: un
**service** (siempre disponible, escala a 0 cuando no se usa) que carga el modelo desde el bucket y
responde a peticiones. Se construye desde `cloud/inferencia/`.

El service es **agnóstico al modelo**: carga el que le digas (`MODEL_GCS` por defecto, o el que pases
en cada petición) y lee del `metrics.json` sus clases y su tamaño de entrada. Lo aprovecharemos en el
Paso 8.

> La organización bloquea el acceso público, así que el service queda **privado** y lo llamamos
> autenticados con un id-token (lo hace `clasificar()` por dentro). Es, de hecho, la práctica
> recomendada aunque pudieras abrirlo.""")
code('''%cd cloud/inferencia
!gcloud run deploy {SERVICE} --source . --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --cpu 2 --memory 4Gi --timeout 180 --min-instances 0 \\
  --set-env-vars MODEL_GCS={MODEL_GCS} -q
%cd ../..
print("Service desplegado en:", _service_url())''')
md("Y clasificamos las tres flores de prueba contra **nuestro** modelo, el que acabamos de entrenar:")
code('''clasificar(IMG)
clasificar(f"gs://{BUCKET}/demo/rosa.jpg")
clasificar(f"gs://{BUCKET}/demo/margarita.jpg")''')

# ============================================================ 8 · MODELO PRE-ENTRENADO
md("""## Paso 8 · Servir un modelo pre-entrenado (sin entrenar nada)

Entrenar está bien cuando las clases son tuyas, pero a veces quieres un modelo **grande y ya entrenado**
sin gastar ni un minuto en entrenarlo. Aquí descargamos **MobileNetV2**, entrenado por Google con
**ImageNet** (1000 clases), y lo servimos **en tu mismo Cloud Run** — ojo, esto es muy distinto de la
Vision API: ahí el modelo corre en los servidores de Google; aquí el modelo es un fichero en **tu**
bucket que sirve **tu** service.

La preparación (bajar el modelo y dejarlo servible) la hace Colab, igual que antes subía las imágenes.""")
code('''# Descargar MobileNetV2 (ImageNet) y dejarlo servible en el bucket, como hizo el job con tu CNN
import tensorflow as tf, json

base = tf.keras.applications.MobileNetV2(weights="imagenet")   # 1000 clases, entrada 224x224
inp  = tf.keras.Input(shape=(224, 224, 3))                     # el service le pasa RGB 0-255
x    = tf.keras.layers.Rescaling(1/127.5, offset=-1)(inp)      # preprocesado de MobileNet, dentro del modelo
servible = tf.keras.Model(inp, base(x))
servible.export(f"{PRETRAIN_GCS}/saved_model")                 # SavedModel directo a gs://

# etiquetas de ImageNet + tamaño de entrada -> metrics.json (lo lee el service, igual que el de flores)
ruta = tf.keras.utils.get_file("imagenet_class_index.json",
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
idx = json.load(open(ruta))
clases = [idx[str(i)][1] for i in range(1000)]
with tf.io.gfile.GFile(f"{PRETRAIN_GCS}/metrics.json", "w") as f:
    f.write(json.dumps({"classes": clases, "img_size": 224, "modelo": "MobileNetV2 (ImageNet)"}))
print("Modelo pre-entrenado listo en", PRETRAIN_GCS)''')
md("""Y ahora lo importante: **no desplegamos nada nuevo**. Llamamos al **mismo service** del Paso 7,
pero apuntando al modelo pre-entrenado (`modelo=PRETRAIN_GCS`). La infra de servir es la misma; solo
cambia el modelo.""")
code('''clasificar(IMG, modelo=PRETRAIN_GCS)
clasificar(f"gs://{BUCKET}/demo/rosa.jpg", modelo=PRETRAIN_GCS)
clasificar(f"gs://{BUCKET}/demo/margarita.jpg", modelo=PRETRAIN_GCS)''')

# ============================================================ 9 · INVENTARIO DE MODELOS
md("""## Paso 9 · Inventario de modelos (un mini "registro")

Ahora tienes **dos** modelos en el bucket y, en cuanto entrenes variantes (más épocas, otras clases,
otra versión…), tendrás más. ¿Cómo sabes qué tienes y dónde está cada uno para poder llamarlo?

La clave: cuando guardamos un modelo, **al lado dejamos su ficha** (`metrics.json` con clases, tamaño,
accuracy…). El propio bucket **es** el registro — no hace falta una base de datos aparte. Si listamos
esas fichas, tenemos un **inventario** en forma de tabla (un DataFrame):""")
code('''inventario = registro_modelos()
inventario''')
md("""Eso es, en pequeño, un **model registry**: un catálogo de qué modelos hay, sus métricas y **dónde
viven** (`ruta_gcs`). Con la ruta, llamar a cualquiera es trivial — eliges la fila que quieras y se la
pasas al service. Versionar es tan simple como guardar en `models/flores/v2`, `.../v3`… y aparecerán
como filas nuevas.""")
code('''# Elegir un modelo del inventario por su nombre y clasificar con él
ruta = inventario.set_index("modelo").loc["imagenet", "ruta_gcs"]
print("Uso el modelo:", ruta)
clasificar(IMG, modelo=ruta)''')

# ============================================================ 10 · CIERRE
md("""## Paso 10 · Repaso, costes y limpieza

Este es el mapa de **todo lo que hemos hecho y dónde ha ocurrido cada cosa** — Colab solo daba
órdenes; lo pesado vivió siempre en tu proyecto de Google Cloud:""")
code('''dibujar_arquitectura()''')
md("""Las ideas que se llevan a casa:

- **Colab ≠ Google Cloud.** Colab fue el mando; el cómputo y lo que persiste vive en GCP.
- **JOB vs SERVICE.** El **job** entrena y muere; el **service** queda sirviendo. Misma plataforma
  (Cloud Run), dos modos para dos necesidades.
- **El bucket lo une todo:** imágenes de entrada, modelos de salida, y el "registro" de modelos.
- **La infra de servir es agnóstica al modelo:** el mismo service sirvió tu CNN y MobileNet. Lo
  entrenes tú o lo descargues hecho, **corre en tu infraestructura** (esa es la diferencia con una
  API gestionada como Vision).

**Costes:** con `--min-instances 0` el service no cuesta nada en reposo; el job solo cuesta mientras
entrena. Todo es **CPU, sin GPU**.

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
