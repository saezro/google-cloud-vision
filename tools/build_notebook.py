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
def code(s, plegada=None):
    """Añade una celda de código. Con `plegada="texto"` sale COLAPSADA en Colab.

    Las celdas que solo definen funciones ocupan pantallas enteras y en la charla
    obligan a hacer scroll delante de todo el mundo. Colab pliega una celda si su
    primera línea es `#@title` y su metadata dice `cellView: form`: se ve una sola
    línea con el título y un triángulo para abrirla si alguien pregunta."""
    meta = {}
    if plegada:
        s = f"#@title {plegada}\n{s}"
        meta = {"cellView": "form"}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None,
                  "outputs": [], "source": s.rstrip("\n").splitlines(keepends=True)})

# ============================================================ PORTADA
md("""# Entrenar y servir tu propio modelo en Google Cloud

De un dataset de imágenes a un modelo en producción, **todo en tu nube**:

1. **Entrenas** una CNN desde cero con tus propias clases, en un **job de Cloud Run**.
2. La **sirves** como una API en un **service de Cloud Run** y haces inferencia sobre fotos nuevas.

El modelo es **tuyo** y corre en **tu** infraestructura: tú lo entrenas, tú lo despliegas, tú lo
llamas. Nada pre-hecho ni externo.

**Cómo va:** dale al play de arriba a abajo. Es idempotente (si algo ya existe, no rompe).

> La teoría (qué es GCP, IAM, **job vs service**, qué es una CNN) va en las slides. Aquí, lo justo
> para seguir el código. Recordatorio de bolsillo: **Colab = el mando; GCP = el cómputo. Un JOB
> entrena y muere; un SERVICE se queda sirviendo.** Damos por hecho un proyecto con **billing**.""")

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

md("""**El resto de la config se deriva del proyecto** (bucket, service account, nombres). No hace
falta tocar nada; si quisieras, es aquí.""")
code('''PROJECT       = _dd.value                               # el que elegiste arriba
REGION        = "europe-west4"                           # región con GPU L4 (bucket + Cloud Run)
BUCKET        = f"{PROJECT}-imagenes"                    # nombre del bucket (único por proyecto)

MODEL_DIR     = "models/flores102"                       # A · tu CNN entrenada desde cero
MODEL_GCS     = f"gs://{BUCKET}/{MODEL_DIR}"             # URI de tu modelo
TRANSFER_DIR  = "models/flores102-transfer"              # B · MobileNetV2 re-entrenada a tus clases
TRANSFER_GCS  = f"gs://{BUCKET}/{TRANSFER_DIR}"          # URI del modelo por transferencia
PRETRAIN_DIR  = "models/imagenet"                        # C · MobileNet tal cual (1000 clases genéricas)
PRETRAIN_GCS  = f"gs://{BUCKET}/{PRETRAIN_DIR}"          # URI del modelo pre-entrenado

RUNTIME_SA = f"taller-vision-sa@{PROJECT}.iam.gserviceaccount.com"  # SA con la que corre todo
JOB          = "taller-entrenar-flores"                 # Cloud Run job con GPU (entrena)
JOB_TRANSFER = "taller-entrenar-transfer"               # el mismo job, con ARCH=transfer
SERVICE      = "taller-inferencia-flores"               # Cloud Run service con GPU (sirve)

# Imágenes de contenedor YA CONSTRUIDAS (pre-charla). El deploy las usa con --image (~30s, sin build).
REPO    = f"{REGION}-docker.pkg.dev/{PROJECT}/cloud-run-source-deploy"
IMG_JOB = f"{REPO}/taller-entrenar-flores:v4"           # imagen del entrenamiento (CUDA), tag fijo
IMG_SVC = f"{REPO}/taller-inferencia-flores:latest"     # imagen de la inferencia (CUDA)

# --- Configuración de la MÁQUINA (lo que pides a Cloud Run para job y service) ---
GPU_TYPE = "nvidia-l4"                                  # tipo de GPU
CPU      = 4                                            # vCPUs (mínimo 4 con GPU)
MEMORY   = "16Gi"                                       # memoria (mínimo 16Gi con GPU L4)

!gcloud config set project {PROJECT} -q
print("Proyecto activo:", PROJECT)
print("Bucket:", BUCKET, "| región:", REGION)
print("Máquina: 1x", GPU_TYPE, "|", CPU, "vCPU |", MEMORY)''')

# ============================================================ UTILIDADES
md("""## Utilidades

Funciones de apoyo que usamos más abajo (mostrar imágenes del bucket, esperar al entrenamiento,
pintar las gráficas, llamar al modelo servido). **No hace falta leerlo** para seguir el taller —
ejecútala y a otra cosa.""")
code('''import io, json, time, subprocess
import requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
import plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image
from IPython.display import display
!pip -q install google-cloud-storage
from google.cloud import storage

# --- Estilo común de las gráficas -----------------------------------------
# Plotly en vez de matplotlib: en la charla se hace zoom y se pasa el ratón por
# encima para leer valores exactos sin volver a ejecutar nada.
# OJO: en Colab hay que dejar el renderer que él detecta ("colab"). Forzar "notebook"
# deja TODAS las gráficas en blanco, incluido el modelo 3D. Fuera de Colab sí se fuerza,
# para que el .ipynb guardado lleve plotly.js dentro y se vea sin conexión.
try:
    import google.colab            # noqa: F401
except ImportError:
    pio.renderers.default = "notebook"
# Azul / naranja / violeta: separación validada también en daltonismo (protan/deutan/tritan)
# y contraste >= 3:1 contra el fondo, que en proyector es donde se cae todo.
C_SERIE = ["#2a78d6", "#eb6834", "#4a3aa7"]
# Magnitud = un solo tono claro->oscuro. Nunca un arcoíris: el salto de color
# sugiere saltos en los datos que no existen.
C_RAMPA = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
_INK, _INK2, _MUTED, _GRID, _SURF, _EJE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"

def _estilo(fig, titulo=None, alto=430):
    """Aplica el mismo aspecto a todas las gráficas: rejilla discreta, tinta gris, leyenda arriba.

    La leyenda sube por encima de los títulos de los paneles: si se deja a la altura
    por defecto se los come."""
    paneles = bool(fig.layout.annotations)      # make_subplots pone ahí los títulos de panel
    arriba = 120 if (paneles and titulo) else 95 if paneles else 70 if titulo else 45
    fig.update_layout(
        template="none", height=alto, title=titulo,
        paper_bgcolor=_SURF, plot_bgcolor=_SURF,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=_INK2),
        title_font=dict(size=16, color=_INK),
        margin=dict(l=60, r=30, t=arriba, b=60),
        legend=dict(orientation="h", yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    y=1.13 if paneles else 1.02, x=.5, xanchor="center"),
    )
    if paneles:                                  # separar los títulos de panel de la leyenda
        for a in fig.layout.annotations:
            a.font = dict(size=14, color=_INK)
    fig.update_xaxes(showgrid=False, linecolor=_EJE, ticks="outside", tickcolor=_EJE, color=_MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False,
                     linecolor="rgba(0,0,0,0)", color=_MUTED)
    return fig

def _barras_redondeadas(fig):
    try: fig.update_layout(barcornerradius=4)          # plotly >= 5.19
    except Exception: pass
    return fig

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
    h = m["history"]
    top5 = m.get("val_top5")
    print(f"clases: {len(m['classes'])}  |  val_accuracy (top-1): {m['val_accuracy']*100:.1f}%"
          + (f"  |  top-5: {top5*100:.1f}%" if top5 else ""))
    ep = list(range(1, len(h["accuracy"]) + 1))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.11,
                        subplot_titles=("Accuracy (acierto)", "Loss (error)"))
    for col, (k_tr, k_va) in enumerate([("accuracy", "val_accuracy"), ("loss", "val_loss")], start=1):
        for i, (k, nom) in enumerate([(k_tr, "entrenamiento"), (k_va, "validación")]):
            fig.add_trace(go.Scatter(
                x=ep, y=h[k], name=nom, legendgroup=nom, showlegend=(col == 1),
                mode="lines+markers", line=dict(color=C_SERIE[i], width=2), marker=dict(size=8),
                hovertemplate=f"<b>{nom}</b>: %{{y:.3f}}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="época", row=1, col=col)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    _estilo(fig, alto=440).update_layout(hovermode="x unified").show()
    pc = sorted(m["accuracy_por_clase"].items(), key=lambda kv: kv[1], reverse=True)
    print("mejores clases:", [f"{k} {v:.0%}" for k, v in pc[:5]])
    print("peores clases: ", [f"{k} {v:.0%}" for k, v in pc[-5:]])
    return m

def _service_url():
    return subprocess.run(["gcloud", "run", "services", "describe", SERVICE, "--region", REGION,
                           "--format=value(status.url)"], capture_output=True, text=True).stdout.strip()

def _token():
    """Token de identidad para llamar al service. Si sale vacío, el 401 luego no se entiende."""
    p = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True)
    tok = p.stdout.strip()
    if not tok:
        raise RuntimeError("gcloud no ha devuelto token de identidad.\\n"
                           f"   {p.stderr.strip()[:300]}\\n"
                           "   En Colab: reejecuta la celda de autenticación. En local: "
                           "gcloud auth login")
    return tok

def _pedir(url, payload, timeout=180):
    """POST al service. Si la respuesta no es un JSON, dice POR QUÉ en vez de reventar
    con un JSONDecodeError, que en directo no dice nada."""
    r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {_token()}"},
                      timeout=timeout)
    if r.status_code != 200 or "json" not in r.headers.get("content-type", "").lower():
        pista = ("el token de identidad no vale para este service"
                 if r.status_code in (401, 403) else
                 "el service ha fallado sirviendo el modelo; mira los logs de Cloud Run"
                 if r.status_code >= 500 else "respuesta inesperada")
        raise RuntimeError(f"El service respondió {r.status_code} y no un JSON ({pista}).\\n"
                           f"   {r.text[:300]}")
    return r.json()

def clasificar(uri, modelo=None):
    """Llama al modelo servido en Cloud Run. `modelo` elige cuál (por defecto, el del service).

    El service es agnóstico al modelo: si le pasas `model_gcs`, sirve ese. Así el MISMO
    endpoint clasifica con tu CNN o con el modelo pre-entrenado, según a cuál apuntes."""
    payload = {"image_gcs": uri}
    if modelo:
        payload["model_gcs"] = modelo
    d = _pedir(f"{_service_url()}/predict", payload)
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
    """CNN en 3D **interactivo** (gíralo y haz zoom con el ratón). Cada bloque es el volumen de datos
    que sale de esa capa: alto×ancho = mapa espacial, grosor = nº de canales."""
    import plotly.graph_objects as go
    colores = {"Entrada": "#9AA7B0", "Conv2D": "#4C72B0", "MaxPooling2D": "#55A868",
               "GlobalAveragePooling2D": "#8172B3", "Dense": "#DD8452"}
    # caja de entrada + solo las capas que cambian de forma (saltamos normalizado y aumento de datos)
    e = tuple(model.layers[0].output.shape)
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
    # triangulación de un cubo (vértices unitarios -> 12 triángulos)
    Vx = [0,0,1,1,0,0,1,1]; Vy = [0,1,1,0,0,1,1,0]; Vz = [0,0,0,0,1,1,1,1]
    I = [7,0,0,0,4,4,6,6,4,0,3,2]; J = [3,4,1,2,5,6,5,2,0,1,6,3]; K = [0,7,2,3,6,7,1,1,5,5,7,6]
    fig = go.Figure(); x = 0.0; lx, ly, lz, lt = [], [], [], []; vistos = set()
    for nombre, h, w, c, etiq in cajas:
        dy, dz = w/8.0, h/8.0; dx = max(c/16.0, 0.3)
        X = [x + dx*t for t in Vx]
        Y = [-dy/2 + dy*t for t in Vy]
        Z = [-dz/2 + dz*t for t in Vz]
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=colores.get(nombre, "#BBB"),
                                opacity=0.92, flatshading=True, name=nombre, legendgroup=nombre,
                                showlegend=(nombre not in vistos),
                                hovertext=f"{nombre}: {etiq}", hoverinfo="text"))
        vistos.add(nombre)
        lx.append(x + dx/2); ly.append(0); lz.append(dz/2 + 1.0); lt.append(etiq)
        x += dx + 2.5
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt, showlegend=False,
                               textposition="top center", textfont=dict(size=11)))
    fig.update_layout(title="Arquitectura de la CNN — gírala con el ratón",
                      height=540, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", y=0),
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False), aspectmode="data",
                                 camera=dict(eye=dict(x=1.5, y=-1.7, z=0.7))))
    fig.show()

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
         ["demo/                  imágenes de prueba", "models/flores102   tu CNN entrenada",
          "models/imagenet    MobileNet descargado"], fc="#EAF0F7")
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

def coste_estimado():
    """ESTIMACIÓN de lo que ha costado la demo (NO es la factura real). El grueso es el job en GPU;
    el storage es céntimos y el service escala a 0. Precios públicos aproximados de europe-west4 (USD).
    La factura exacta sale en la consola de Facturación, con horas de retraso."""
    from datetime import datetime
    P_GPU, P_VCPU, P_MEM, P_STORE_MES = 0.000233, 0.0000240, 0.0000025, 0.020  # $/s y $/GiB·mes
    mem_gib = int("".join(ch for ch in MEMORY if ch.isdigit()))                # "16Gi" -> 16
    filas = []
    # 1) Jobs: sumar la duración de TODAS las ejecuciones (incluidas las fallidas, también cuestan)
    seg, n = 0.0, 0
    for job in (JOB, JOB_TRANSFER):
        out = subprocess.run(["gcloud", "run", "jobs", "executions", "list", "--job", job,
            "--region", REGION, "--format=value(status.startTime,status.completionTime)"],
            capture_output=True, text=True).stdout.strip().splitlines()
        for ln in out:
            p = ln.split()
            if len(p) == 2:
                t0 = datetime.fromisoformat(p[0].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(p[1].replace("Z", "+00:00"))
                seg += max((t1 - t0).total_seconds(), 0); n += 1
    c_job = seg * (P_GPU + CPU * P_VCPU + mem_gib * P_MEM)
    filas.append(("Jobs de entrenamiento (GPU L4)", f"{seg/60:.1f} min · {n} ejec.", round(c_job, 3)))
    # 2) Storage del bucket (prorrateado ~1 día)
    gib = sum((b.size or 0) for b in _sc.list_blobs(BUCKET)) / (1024**3)
    filas.append(("Storage (bucket, ~1 día)", f"{gib:.2f} GiB", round(gib * P_STORE_MES / 30, 3)))
    # 3) Service: escala a 0, solo ms por petición -> despreciable
    filas.append(("Service inferencia (GPU, escala a 0)", "~ms/petición", 0.0))
    df = pd.DataFrame(filas, columns=["recurso", "uso", "coste_usd_aprox"])
    print("ESTIMACIÓN (no es la factura real; precios aprox. europe-west4, USD)")
    print(f"TOTAL ESTIMADO ≈ ${df['coste_usd_aprox'].sum():.2f}")
    return df

def _metrics(model_dir):
    """Lee el metrics.json de un modelo del bucket. `model_dir` es 'models/loquesea'."""
    return json.loads(_bucket().blob(f"{model_dir.strip('/')}/metrics.json").download_as_text())

def comparar_modelos(*model_dirs):
    """Tabla comparativa de varios modelos: cuánto aciertan, cuánto ocupan y cuánto costó entrenarlos.

    Todos se evalúan sobre el MISMO conjunto de validación, así que los números son comparables."""
    filas = []
    for d in model_dirs:
        try:
            m = _metrics(d)
        except Exception:
            print(f"(sin metrics.json: {d})"); continue
        seg = m.get("train_seconds")
        filas.append({
            "modelo": d.split("/", 1)[-1],
            "arquitectura": m.get("arch", "pre-entrenado"),
            "top-1": f"{m['val_accuracy']*100:.1f}%" if m.get("val_accuracy") else "—",
            "top-5": f"{m['val_top5']*100:.1f}%" if m.get("val_top5") else "—",
            "params": f"{m['n_params']/1e6:.2f} M" if m.get("n_params") else "—",
            "entrenables": f"{m['n_params_entrenables']/1e6:.2f} M" if m.get("n_params_entrenables") else "—",
            "épocas": m.get("epochs", "—"),
            "entreno": f"{seg/60:.1f} min" if seg else "—",
            "img": m.get("img_size"),
        })
    return pd.DataFrame(filas)

def curvas_comparadas(*model_dirs):
    """Superpone las curvas de aprendizaje. Se ve de un vistazo que el transfer arranca ya alto."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.11,
                        subplot_titles=("Accuracy en validación", "Loss en validación"))
    for i, d in enumerate(model_dirs):
        try:
            h = _metrics(d)["history"]
        except Exception:
            continue
        etq = d.split("/", 1)[-1]
        color = C_SERIE[i % len(C_SERIE)]           # el color va con el modelo, no con su puesto
        ep = list(range(1, len(h["val_accuracy"]) + 1))
        for col, k in enumerate(["val_accuracy", "val_loss"], start=1):
            fig.add_trace(go.Scatter(
                x=ep, y=h[k], name=etq, legendgroup=etq, showlegend=(col == 1),
                mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8),
                hovertemplate=f"<b>{etq}</b>: %{{y:.3f}}<extra></extra>"), row=1, col=col)
    for col in (1, 2):
        fig.update_xaxes(title_text="época", row=1, col=col)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    _estilo(fig, alto=460).update_layout(hovermode="x unified").show()

def informe_por_clase(model_dir, n=10):
    """Precision, recall y F1 por clase, derivados de la matriz de confusión.

    accuracy sola engaña: un modelo puede acertar mucho de media y fallar sistemáticamente
    en las clases que a ti te importan. Esto es lo que hay que mirar."""
    m = _metrics(model_dir); C = np.array(m["matriz_confusion"]); clases = m["classes"]
    tp = np.diag(C).astype(float)
    soporte = C.sum(axis=1)                    # cuántas imágenes reales hay de cada clase
    predichas = C.sum(axis=0)                  # cuántas veces el modelo dijo esa clase
    precision = np.divide(tp, predichas, out=np.zeros_like(tp), where=predichas > 0)
    recall = np.divide(tp, soporte, out=np.zeros_like(tp), where=soporte > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    df = pd.DataFrame({"clase": clases, "precision": precision.round(3), "recall": recall.round(3),
                       "f1": f1.round(3), "soporte": soporte}).sort_values("f1", ascending=False)
    print(f"{model_dir}  ·  F1 macro (media sin ponderar): {f1.mean():.3f}")
    print(f"\\nLas {n} clases que MEJOR reconoce:"); display(df.head(n).reset_index(drop=True))
    print(f"\\nLas {n} que PEOR — aquí es donde se pierde dinero en producción:")
    display(df.tail(n).reset_index(drop=True))
    # En barras se ve de un golpe si el fallo es de precision (dice esa clase y no lo es)
    # o de recall (era esa clase y no la vio). No es lo mismo y no se arregla igual.
    # Ojo: en un modelo malo las N peores están TODAS a cero y el gráfico saldría en blanco,
    # así que se pintan las peores que aún puntúan algo y las de cero se cuentan aparte.
    ceros = df[df["f1"] == 0]
    peor = df[df["f1"] > 0].tail(n).iloc[::-1]
    modelo = model_dir.split("/")[-1]
    if peor.empty:
        print(f"({modelo}: ninguna clase con F1 > 0, no hay nada que pintar)")
        return df
    fig = go.Figure()
    for i, (col, nom) in enumerate([("precision", "precision"), ("recall", "recall"), ("f1", "F1")]):
        fig.add_trace(go.Bar(
            x=peor["clase"], y=peor[col], name=nom, marker_color=C_SERIE[i],
            hovertemplate=f"<b>%{{x}}</b><br>{nom}: %{{y:.2f}}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(tickangle=-40, automargin=True)
    titulo = f"Las {len(peor)} clases peor reconocidas (de las que acierta alguna) — {modelo}"
    _barras_redondeadas(_estilo(fig, titulo, alto=490))
    fig.update_layout(bargap=.28, bargroupgap=.06, hovermode="x unified")
    if len(ceros):
        fig.add_annotation(
            x=0, y=1.06, xref="paper", yref="paper", showarrow=False, xanchor="left",
            font=dict(size=12, color=_MUTED),
            text=f"Además hay {len(ceros)} clases con F1 = 0: el modelo no acierta ni una.")
    fig.show()
    return df

def matriz_confusion(model_dir, n=18):
    """Heatmap de la matriz de confusión, quedándonos con las N clases que más se confunden.

    Con 102 clases la matriz entera no se lee. Nos quedamos con las peores: la diagonal es
    lo que acierta, todo lo que se sale de ella es con QUÉ lo confunde."""
    m = _metrics(model_dir); C = np.array(m["matriz_confusion"]); clases = m["classes"]
    acierto = np.divide(np.diag(C), np.maximum(C.sum(axis=1), 1))
    peores = np.argsort(acierto)[:n]                    # las n clases con menos acierto
    sub = C[np.ix_(peores, peores)].astype(float)
    filas = np.maximum(sub.sum(axis=1, keepdims=True), 1)
    prop = sub / filas
    etq = [clases[i][:22] for i in peores]
    # El número de imágenes va escrito en la celda: el color da la intensidad, el texto el dato exacto
    txt = [[str(int(v)) if v else "" for v in fila] for fila in sub]
    fig = go.Figure(go.Heatmap(
        z=prop, x=etq, y=etq, zmin=0, zmax=1, colorscale=C_RAMPA, xgap=2, ygap=2,
        text=txt, texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title=dict(text="proporción de<br>la clase real", side="right"),
                      thickness=12, outlinewidth=0, tickformat=".0%", len=.8),
        hovertemplate="ERA <b>%{y}</b><br>predijo <b>%{x}</b><br>"
                      "%{text} imágenes · %{z:.0%} de la clase<extra></extra>"))
    # scaleanchor: celdas cuadradas. Sin esto salen rectángulos alargados y la diagonal engaña.
    fig.update_yaxes(autorange="reversed", title_text="lo que ERA de verdad", showgrid=False,
                     scaleanchor="x", constrain="domain")
    fig.update_xaxes(tickangle=-90, title_text="lo que PREDIJO", constrain="domain")
    _estilo(fig, f"Confusión en las {n} clases peor reconocidas — {model_dir.split('/')[-1]}",
            alto=700).update_layout(margin=dict(l=180, r=30, t=80, b=180)).show()

def benchmark_latencia(uri, modelos, repeticiones=12):
    """Mide la latencia REAL del service con cada modelo y estima el coste por 1.000 inferencias.

    `modelos` es un dict {etiqueta: ruta_gcs}. La primera llamada de cada modelo carga el
    SavedModel en la GPU (arranque en frío) y se descarta: no representa el régimen normal."""
    P_GPU, P_VCPU, P_MEM = 0.000233, 0.0000240, 0.0000025      # $/segundo en europe-west4
    mem_gib = int("".join(ch for ch in MEMORY if ch.isdigit()))
    coste_seg = P_GPU + CPU * P_VCPU + mem_gib * P_MEM
    tok = _token()
    url = f"{_service_url()}/predict"
    filas = []
    for etq, ruta in modelos.items():
        ms = []
        for i in range(repeticiones + 1):
            t0 = time.time()
            r = requests.post(url, json={"image_gcs": uri, "model_gcs": ruta},
                              headers={"Authorization": f"Bearer {tok}"}, timeout=180)
            dt = (time.time() - t0) * 1000
            if r.status_code != 200:      # medir la latencia de un error no significa nada
                raise RuntimeError(f"{etq}: el service respondió {r.status_code} — {r.text[:200]}")
            if i == 0:
                frio = dt          # la primera carga el modelo: fuera del cálculo
                continue
            ms.append(dt)
        ms = np.array(ms)
        filas.append({
            "modelo": etq,
            "arranque en frío": f"{frio/1000:.1f} s",
            "p50": f"{np.percentile(ms, 50):.0f} ms",
            "p95": f"{np.percentile(ms, 95):.0f} ms",
            "$/1.000 inferencias": round(np.percentile(ms, 50) / 1000 * coste_seg * 1000, 4),
        })
    print("Latencia extremo a extremo (incluye la red desde Colab, no solo la GPU).")
    df = pd.DataFrame(filas)
    # p50 = lo que nota el usuario normal; p95 = lo que nota el que tiene mala suerte.
    # La barra clara detrás es el arranque en frío, a otra escala: por eso va en su propio gráfico.
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.16,
                        subplot_titles=("Latencia por petición", "Coste por 1.000 inferencias"))
    etqs = [f["modelo"] for f in filas]
    for i, k in enumerate(["p50", "p95"]):
        fig.add_trace(go.Bar(
            x=etqs, y=[float(f[k].split()[0]) for f in filas], name=k, marker_color=C_SERIE[i],
            hovertemplate=f"<b>%{{x}}</b><br>{k}: %{{y:.0f}} ms<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(
        x=etqs, y=[f["$/1.000 inferencias"] for f in filas], name="$ / 1.000", showlegend=False,
        marker_color=C_SERIE[2], text=[f"${f['$/1.000 inferencias']:.3f}" for f in filas],
        textposition="outside", textfont=dict(color=_INK2),
        hovertemplate="<b>%{x}</b><br>$%{y:.4f} por 1.000<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title_text="milisegundos", row=1, col=1)
    fig.update_yaxes(title_text="USD", tickprefix="$", row=1, col=2)
    _barras_redondeadas(_estilo(fig, alto=440))
    fig.update_layout(bargap=.3, bargroupgap=.06).show()
    return df

print("Utilidades listas")''',
     plegada="Utilidades del taller · ejecútala y sigue (ábrela solo si quieres ver el código)")


# ============================================================ 2 · APIs
md("""## Paso 2 · Activar las APIs del proyecto

Cada servicio se activa **por proyecto**. Encendemos los que usaremos: Storage, Cloud Run, Cloud
Build, Artifact Registry e IAM.""")
code('''!gcloud services enable \\
  storage.googleapis.com \\
  run.googleapis.com cloudbuild.googleapis.com \\
  artifactregistry.googleapis.com iam.googleapis.com -q
print("APIs activadas")''')

# ============================================================ 3 · IAM
md("""## Paso 3 · IAM — quién puede hacer qué

Montamos dos cosas: (3.1) una **service account de runtime** con la que corren job y service (sin
claves en el código), y (3.2) los **permisos de build**. Dos detalles clave (en las slides está el
porqué):

- `serviceAccountUser` = **"actuar como"** la SA: sin él, el deploy falla. Es el error nº1.
- El build corre como la **SA de Compute** (cambio de 2024), no la `@cloudbuild`. Damos permisos a
  **las dos** por si acaso. El síntoma despista: habla de `storage.objects.get`.""")
code('''# 3.1 — Crear la service account con la que correrán job y service
!gcloud iam service-accounts create taller-vision-sa \\
  --display-name="Taller Vision runtime" 2>/dev/null || echo "(ya existe)"

# ...y darle acceso al bucket y a consumir APIs
for ROLE in ["roles/storage.admin", "roles/serviceusage.serviceUsageConsumer"]:
    !gcloud projects add-iam-policy-binding {PROJECT} \\
      --member="serviceAccount:{RUNTIME_SA}" --role={ROLE} --condition=None -q > /dev/null
print("Service account de runtime con permisos")''')
code('''# 3.2 — Permisos de despliegue para quien construye (las DOS posibles SA de build)
_pnum = !gcloud projects describe {PROJECT} --format="value(projectNumber)"
PNUM = _pnum[0].strip()
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

El **bucket** es el almacén (imágenes ahora, modelos después). Lo creamos y subimos las fotos de
prueba del repo.""")
code('''# Crear el bucket (idempotente). Si ya existía -> mensaje en verde, sin error rojo.
import subprocess
def _gc(*args):
    return subprocess.run(["gcloud", *args], capture_output=True, text=True)

VERDE, ROJO, FIN = "\\033[92m", "\\033[91m", "\\033[0m"
_r = _gc("storage", "buckets", "create", f"gs://{BUCKET}",
         f"--location={REGION}", "--uniform-bucket-level-access", "-q")
if _r.returncode == 0:
    print(f"{VERDE}✓ Bucket creado: gs://{BUCKET}{FIN}")
elif _gc("storage", "buckets", "describe", f"gs://{BUCKET}", "--format=value(name)").returncode == 0:
    print(f"{VERDE}✓ El bucket ya estaba creado: gs://{BUCKET}{FIN}")
else:
    print(f"{ROJO}✗ Error creando el bucket:{FIN}\\n{_r.stderr.strip()}")''')
code('''# Subir las imágenes de prueba del repo (carpeta imagenes/) a gs://BUCKET/demo/
import glob
for ruta in sorted(glob.glob("imagenes/*.jpg")):
    !gcloud storage cp "{ruta}" gs://{BUCKET}/demo/ -q
IMG = f"gs://{BUCKET}/demo/flor.jpg"   # imagen por defecto para los ejemplos de abajo
print("Subidas:", sorted(glob.glob("imagenes/*.jpg")))''')
md("Y las vemos *desde el bucket* (se descargan de GCS, no son las locales):")
code('''ver_bucket()''')

# ============================================================ IMÁGENES (precompiladas)
md("""## Las imágenes de los contenedores (ya construidas)

El job y el service son **contenedores**. Su imagen (con `tensorflow[and-cuda]` para GPU) se construye
**una sola vez** con Cloud Build a partir de `cloud/entrenamiento/` y `cloud/inferencia/`, y se guarda
en **Artifact Registry**. Así el `deploy` de después es **instantáneo** (`--image`, ~30 s) en vez de
esperar ~10 min de build.

Se construyen así (es lento, **ya está hecho** — en la charla no se re-ejecuta):

```
gcloud builds submit cloud/entrenamiento --tag IMG_JOB --region REGION
gcloud builds submit cloud/inferencia    --tag IMG_SVC --region REGION
```

Comprobamos que las imágenes existen:""")
code('''print("Job   :", IMG_JOB)
print("Service:", IMG_SVC)
!gcloud artifacts docker images list {REPO} --include-tags --filter="tags:latest" --format="value(package,tags)" 2>/dev/null''')

# ============================================================ 5 · ENTRENAR
md("""## Paso 5 · Entrenar tu propia CNN desde cero en Cloud Run (job, con GPU)

El corazón del taller: **entrenamos una CNN desde cero** sobre **Oxford Flowers 102** (102 clases de
flor, ~8.000 imágenes), en una **GPU NVIDIA L4**. Antes de lanzarlo, veamos **qué** vamos a entrenar.""")

md("""### Hiperparámetros del entrenamiento

Estos son los mandos del entrenamiento. Cámbialos aquí (es lo que define cómo y cuánto entrena):
- `EPOCHS`: cuántas pasadas al dataset. Más = mejor (hasta cierto punto) pero más lento.
- `IMG_SIZE`: a qué tamaño se redimensiona cada imagen de entrada de la CNN.""")
code('''EPOCHS   = 15    # épocas (en GPU L4, ~14 s/época → ~3-4 min con 15)
IMG_SIZE = 180   # lado de la imagen de entrada (px)''')

md("""### La arquitectura de la CNN

La red que entrenamos (la teoría de conv/pool/dropout, en las slides). Mostramos **su código real**,
el mismo que usa el job — lo importamos del repo, sin copiar nada:""")
code('''# Importamos la arquitectura REAL del job y mostramos su código de construcción
import sys, inspect
sys.path.insert(0, "cloud/entrenamiento")
from train import construir_modelo

print(inspect.getsource(construir_modelo))''')
md("La instanciamos (102 clases, `IMG_SIZE`×`IMG_SIZE`) y miramos su resumen — capas, formas y nº de parámetros:")
code('''modelo_demo = construir_modelo(n_clases=102, img=IMG_SIZE)
modelo_demo.summary()''')
md("""Y en **3D interactivo** (gíralo y haz zoom con el ratón para ver los ángulos): cada bloque es el
volumen de datos que sale de esa capa. El embudo típico de una CNN — el **mapa espacial encoge**
mientras la **profundidad de canales crece** (32→64→128→256), hasta que el clasificador lo reduce a
102 probabilidades.""")
code('''dibujar_cnn_3d(modelo_demo)''')

md("""> El resto del entrenamiento (cargar Flores-102, el bucle de entreno, guardar el modelo) vive en
> **`cloud/entrenamiento/train.py`** — ábrelo en el explorador de archivos de Colab para verlo entero.

### Crear el job y lanzar el entrenamiento (en GPU)

Creamos el **job con GPU** (`--gpu 1 --gpu-type nvidia-l4`) **desde la imagen ya construida**
(`--image`, sin esperar al build) y lo ejecutamos en segundo plano (`--async`). TensorFlow ve la GPU
solo.

> **Coste:** un **job no cuesta nada en reposo** — arranca, entrena, **libera la GPU y muere**. Solo
> pagas los minutos de entrenamiento. El `--task-timeout` es un tope de seguridad.
> En la sesión el modelo ya está en el bucket, así que el Paso 6 no espera.""")
code('''!gcloud run jobs deploy {JOB} --image {IMG_JOB} --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --gpu 1 --gpu-type {GPU_TYPE} --no-gpu-zonal-redundancy \\
  --cpu {CPU} --memory {MEMORY} --task-timeout 3600 --max-retries 0 \\
  --set-env-vars BUCKET={BUCKET},MODEL_DIR={MODEL_DIR},EPOCHS={EPOCHS},IMG_SIZE={IMG_SIZE} -q
!gcloud run jobs execute {JOB} --region {REGION} --async
print("Entrenando en GPU. Llama a esperar_modelo() cuando quieras el resultado.")''')

# ============================================================ 6 · STATS
md("""## Paso 6 · Ver cómo ha aprendido el modelo

El job deja un `metrics.json` en el bucket. Esperamos a que esté (si ya estaba, sigue al momento) y
pintamos las curvas de accuracy y loss.""")
code('''esperar_modelo()
m = stats()''')

# ============================================================ 7 · INFERIR
md("""## Paso 7 · Servir tu modelo en Cloud Run e inferir (en GPU)

Creamos un **service con GPU** (`--gpu 1 --gpu-type nvidia-l4`) **desde la imagen ya construida**
(`--image`). Es **agnóstico al modelo**: carga el que le digas y lee sus clases y tamaño del
`metrics.json` — lo aprovecharemos en el Paso 8.

> **Coste — el apagado automático es CLAVE con GPU.** Va con `--min-instances 0`: cuando nadie lo usa,
> Cloud Run **apaga la instancia y deja de cobrar la GPU**. Sin esto, una GPU encendida cuesta cada
> hora. **Lo dejamos en 0 también para la charla**: la primera inferencia paga un **arranque en frío**
> de ~5 s (cargar el modelo en GPU) y las siguientes van a milisegundos. Así no pagas GPU parada.
> `--max-instances` pone un techo de gasto.
>
> Queda **privado** (la org bloquea el acceso público): se llama con un id-token, lo hace
> `clasificar()` por dentro.""")
md("""> El código del service —una API FastAPI que carga el SavedModel de GCS y responde en `/predict`—
> está en **`cloud/inferencia/main.py`**. Ábrelo para explicarlo por dentro.""")
code('''!gcloud run deploy {SERVICE} --image {IMG_SVC} --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --gpu 1 --gpu-type {GPU_TYPE} --no-gpu-zonal-redundancy \\
  --cpu {CPU} --memory {MEMORY} --timeout 180 --min-instances 0 --max-instances 1 \\
  --set-env-vars MODEL_GCS={MODEL_GCS} -q
print("Service (GPU) desplegado en:", _service_url())''')
md("Y clasificamos las tres flores de prueba contra **nuestro** modelo, el que acabamos de entrenar:")
code('''clasificar(IMG)
clasificar(f"gs://{BUCKET}/demo/rosa.jpg")
clasificar(f"gs://{BUCKET}/demo/margarita.jpg")''')

# ============================================================ 8 · MODELO PRE-ENTRENADO
md("""## Paso 8 · Servir un modelo pre-entrenado (sin entrenar nada)

A veces quieres un modelo **grande y ya entrenado** sin entrenarlo tú. Descargamos **MobileNetV2**
(ImageNet, 1000 clases) y lo servimos en **tu mismo Cloud Run** (no es la Vision API: aquí el modelo
es un fichero en **tu** bucket que sirve **tu** service). La descarga la hace Colab, como con las
imágenes.""")
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

# ============================================================ 9 · COMPARAR MODELOS
md("""## Paso 9 · Comparar modelos con datos, no con impresiones

Hasta aquí hemos entrenado un modelo y servido otro. La pregunta de verdad, la que se hace cualquiera
que vaya a poner esto en producción, es otra: **¿cuál elijo, y cómo lo justifico?**

Tenemos tres candidatos sobre el **mismo dataset** y el **mismo service**, que es la única forma de
que los números sean comparables:

| | Modelo | Qué es | Qué esperar |
|---|---|---|---|
| **A** | `flores102` | CNN **desde cero**: aprende a ver partiendo de nada | acierto modesto, entreno largo |
| **B** | `flores102-transfer` | **MobileNetV2** de ImageNet congelada + cabeza nueva de 102 clases | mucho más acierto, entreno corto |
| **C** | `imagenet` | MobileNet **tal cual**, sin tocar | falla: sus 1.000 clases no son tus flores |

La comparación A vs B es la decisión de arquitectura más rentable del taller, y C es el recordatorio
de que un modelo excelente en abstracto es inútil si no responde **tu** pregunta.

### Entrenar el modelo B (transfer learning)

Es **el mismo job y la misma imagen**: solo cambia `ARCH=transfer`. Corre en segundo plano y en la
sesión ya está entrenado, así que no esperamos.""")
code('''!gcloud run jobs deploy {JOB_TRANSFER} --image {IMG_JOB} --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --gpu 1 --gpu-type {GPU_TYPE} --no-gpu-zonal-redundancy \\
  --cpu {CPU} --memory {MEMORY} --task-timeout 3600 --max-retries 0 \\
  --set-env-vars BUCKET={BUCKET},MODEL_DIR={TRANSFER_DIR},ARCH=transfer -q
!gcloud run jobs execute {JOB_TRANSFER} --region {REGION} --async
print("Entrenando el modelo por transferencia. El de la sesión ya está en el bucket.")''')

md("""### La tabla que enseñarías en una reunión

Acierto, tamaño y coste de entrenamiento, lado a lado.""")
code('''comparar_modelos(MODEL_DIR, TRANSFER_DIR)''')

md("""Fíjate en la columna **entrenables**: el modelo por transferencia tiene muchos más parámetros
en total, pero entrena solo una fracción — el resto viene aprendido. Por eso cuesta menos GPU y
acierta más. Menos cómputo y mejor resultado a la vez.

### Cómo aprendió cada uno""")
code('''curvas_comparadas(MODEL_DIR, TRANSFER_DIR)''')

md("""El de transferencia **arranca ya alto en la primera época**: no está aprendiendo a ver, solo a
nombrar flores. La CNN desde cero tiene que aprender antes qué es un borde.

### Dónde falla exactamente

La accuracy media es un número tranquilizador y poco útil. Lo que importa en producción es **qué
clases falla y con qué las confunde** — ahí es donde un error cuesta dinero.""")
code('''matriz_confusion(MODEL_DIR)''')
md("""Cada fila es una clase real y cada columna lo que el modelo predijo: la diagonal es acierto y
todo lo que se sale de ella es una confusión concreta, con nombre y apellidos.""")
code('''informe_por_clase(MODEL_DIR)''')
md("""**Precision** = de las veces que dijo esta clase, cuántas acertó (mide falsos positivos).
**Recall** = de las que había de verdad, cuántas encontró (mide lo que se le escapa). Según el caso
de uso te importa una u otra: en un control de calidad prefieres no dejar pasar un defecto (recall);
en una alerta que despierta a alguien de noche, prefieres no dar falsas alarmas (precision).

Y lo mismo para el modelo por transferencia, para ver si además de acertar más, falla mejor:""")
code('''informe_por_clase(TRANSFER_DIR)''')

md("""### Los tres, sobre la misma foto

Mismo endpoint, misma imagen, tres modelos. Solo cambia a cuál apuntamos.""")
code('''clasificar(IMG, modelo=MODEL_GCS)      # A · CNN desde cero
clasificar(IMG, modelo=TRANSFER_GCS)   # B · transfer learning
clasificar(IMG, modelo=PRETRAIN_GCS)   # C · MobileNet de ImageNet, sin adaptar''')
md("""C contesta con aplomo una clase de ImageNet que no es lo que le preguntas. **Un modelo no sabe
que no sabe**: por eso se evalúa contra tus datos y no contra la reputación del modelo.

### Lo que de verdad pregunta un cliente: cuánto tarda y cuánto cuesta""")
code('''benchmark_latencia(IMG, {"A · desde cero": MODEL_GCS,
                          "B · transfer": TRANSFER_GCS,
                          "C · imagenet": PRETRAIN_GCS})''')
md("""El **arranque en frío** es el precio de escalar a cero: la primera petición carga el modelo en
la GPU. Si tu tráfico es a ráfagas, compensa de sobra; si necesitas respuesta inmediata siempre,
subes `--min-instances` a 1 y pagas la GPU parada. Esa es la decisión, y ahora tienes el número
para tomarla en vez de opinar.""")

# ============================================================ 10 · INVENTARIO DE MODELOS
md("""## Paso 10 · Inventario de modelos (un mini "registro")

Ya tienes **tres** modelos, y con el tiempo más. Cada uno guarda su ficha (`metrics.json`) al lado, así
que **el propio bucket es el registro**: listando esas fichas tienes un inventario (un DataFrame), sin
base de datos aparte.""")
code('''inventario = registro_modelos()
inventario''')
md("""Un **model registry** en pequeño: qué hay, sus métricas y **dónde vive cada uno** (`ruta_gcs`).
Con esa ruta llamas a cualquiera. Versionar = guardar en `models/flores102/v2`, `v3`… → filas nuevas.""")
code('''# Elegir un modelo del inventario por su nombre y clasificar con él
ruta = inventario.set_index("modelo").loc["imagenet", "ruta_gcs"]
print("Uso el modelo:", ruta)
clasificar(IMG, modelo=ruta)''')

# ============================================================ 11 · CIERRE
md("""## Paso 11 · Repaso, costes y limpieza

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

**Costes — quién paga qué y cuándo (todo en GPU, así que el apagado importa):**

| Recurso | ¿Cuesta en reposo? | Cómo lo controlamos |
|---|---|---|
| **Job (GPU)** | **No.** Arranca, entrena y **muere**. | Solo pagas los minutos de entrenamiento. `--task-timeout` = tope. |
| **Service (GPU)** | **No**, con `--min-instances 0`. | **Se apaga solo** cuando nadie lo usa (la GPU deja de cobrar). `--max-instances` = techo. |
| **Bucket** | Céntimos. | Almacenamiento de imágenes y modelos. |

Las dos GPU solo están encendidas cuando hacen falta: el **job** los minutos que entrena, el
**service** solo mientras atiende peticiones. **El service se deja en `--min-instances 0` también para
la charla**: la primera inferencia paga ~5 s de arranque y ya; así una GPU parada nunca cobra.""")
md("""### ¿Cuánto ha costado todo esto?

Una **estimación** de lo gastado (job en GPU + storage; el service escala a 0). No es la factura real
—esa sale en la consola de Facturación con horas de retraso— pero da el orden de magnitud al momento:""")
code('''coste_estimado()''')
md("""**Limpieza** (se lleva por delante todo lo creado, incluido cualquier resto que cobre):

```
!gcloud projects delete {PROJECT}
```""")

# ============================================================ EXTRA · crear desde cero en vivo
md("""## Extra · Crear un job y un service desde cero, en vivo (si da tiempo)

Para **enseñar la mecánica de crear** un job y un service (con otros nombres, sin tocar los de arriba).
Como van **desde la imagen ya construida** (`--image`), crear los dos tarda **~1 minuto**. Si no da
tiempo, sáltatelo.

> **Cuota de GPU:** tu proyecto tiene un tope de memoria por región. Si ya están vivos el job y el
> service de arriba (16 GiB cada uno), crear estos otros con GPU puede chocar con la cuota. Si pasa,
> borra primero los de arriba (`gcloud run jobs delete ...`, `gcloud run services delete ...`) o pide
> más cuota. Al final hay una celda de limpieza.""")
code('''JOB_DEMO = "demo-entrenar"
SVC_DEMO = "demo-inferencia"

# 1) Crear el JOB (desde la imagen, ~30 s)
!gcloud run jobs deploy {JOB_DEMO} --image {IMG_JOB} --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --gpu 1 --gpu-type {GPU_TYPE} --no-gpu-zonal-redundancy \\
  --cpu {CPU} --memory {MEMORY} --task-timeout 3600 --max-retries 0 \\
  --set-env-vars BUCKET={BUCKET},MODEL_DIR=models/demo,EPOCHS={EPOCHS},IMG_SIZE={IMG_SIZE} -q

# 2) Crear el SERVICE (desde la imagen, ~30-60 s)
!gcloud run deploy {SVC_DEMO} --image {IMG_SVC} --region {REGION} \\
  --service-account {RUNTIME_SA} \\
  --gpu 1 --gpu-type {GPU_TYPE} --no-gpu-zonal-redundancy \\
  --cpu {CPU} --memory {MEMORY} --timeout 180 --min-instances 0 --max-instances 1 \\
  --set-env-vars MODEL_GCS={MODEL_GCS} -q
print("Job y service de demo creados.")''')
md("Opcional: lanzar el entrenamiento del job de demo (~3-4 min en GPU).")
code('''!gcloud run jobs execute {JOB_DEMO} --region {REGION} --async
print("Entrenando (demo) en segundo plano.")''')
md("Limpieza de los recursos de demo (para liberar la cuota de GPU):")
code('''!gcloud run jobs delete {JOB_DEMO} --region {REGION} -q
!gcloud run services delete {SVC_DEMO} --region {REGION} -q
print("Recursos de demo borrados.")''')

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "name": "google-cloud-vision.ipynb"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "google-cloud-vision.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("escrito google-cloud-vision.ipynb con", len(cells), "celdas")
