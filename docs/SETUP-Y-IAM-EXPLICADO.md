# Cómo está montado todo (y por qué) — guion técnico para explicar en el taller

Este documento explica, paso a paso, la infraestructura real que usa el taller, montada en el
proyecto GCP **`tu-proyecto-gcp`** (nombre visible "Google Cloud Vision"). Está pensado para que
puedas **explicarlo en directo**, con foco en la parte que más confunde y más importa: **IAM**.

## 0 · La foto completa

```
Tú (tu-usuario@example.com)  ──crea──►  Proyecto GCP "tu-proyecto-gcp"
                                  ├─  Cloud Storage   bucket: imágenes + modelo
                                  ├─  Vision API       visión gestionada
                                  ├─  Cloud Run JOB     entrena la CNN  (corre COMO una SA)
                                  └─  Cloud Run SERVICE sirve la CNN     (corre COMO una SA)
```

Tres ideas que quiero que la sala se lleve:
1. **Un proyecto** es la unidad de aislamiento y de facturación en GCP. Todo vive dentro de uno.
2. **Para usar una API hay que habilitarla** en ese proyecto (no vienen activas por defecto).
3. **IAM decide quién puede hacer qué.** Sin los permisos correctos, nada funciona — y es donde
   todo el mundo se atasca. Por eso le dedicamos una sección entera.

---

## 1 · El proyecto y la facturación

```bash
gcloud projects create tu-proyecto-gcp --name="Google Cloud Vision" --organization=TU-ORG-ID
gcloud billing projects link tu-proyecto-gcp --billing-account=XXXXXX-XXXXXX-XXXXXX
```

- Un **proyecto** agrupa recursos, permisos y facturación. Lo creamos **separado de producción**
  para no mezclar nada y poder **borrarlo entero** al final (`gcloud projects delete`).
- **Billing**: aunque usemos capa gratuita, GCP exige una cuenta de facturación vinculada para poder
  activar APIs de pago como Vision o Cloud Run.
- Detalle real: el ID **no puede contener "google"** ni espacios → por eso el ID es
  `tu-proyecto-gcp` aunque el nombre visible sea "Google Cloud Vision".

---

## 2 · Activar las APIs

```bash
gcloud services enable storage.googleapis.com vision.googleapis.com \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com iam.googleapis.com
```

Cada servicio de Google Cloud es una **API que se activa por proyecto**. Aquí activamos:
- **storage** (buckets), **vision** (la API de visión),
- **run** (Cloud Run), **cloudbuild** (construir las imágenes de contenedor),
- **artifactregistry** (guardar esas imágenes), **iam** (gestionar permisos).

> Mensaje para la sala: si una llamada falla con `SERVICE_DISABLED`, casi siempre es que falta
> activar la API en *este* proyecto.

---

## 3 · IAM — quién puede hacer qué   (la parte importante)

**IAM = Identity and Access Management.** Es el sistema de permisos de Google Cloud. Se resume en una
frase:

> **Una IDENTIDAD tiene un ROL sobre un RECURSO.**  (quién · puede hacer qué · sobre qué)

### 3.1 · Tipos de identidad

- **Usuario** (`tu-usuario@example.com`): una persona. La usamos para crear el proyecto.
- **Cuenta de servicio / Service Account (SA)** (`...@...iam.gserviceaccount.com`): una identidad
 **para máquinas/código**, no para personas. Los servicios (Cloud Run) **corren COMO una SA**.
- **SAs de Google gestionadas**: Google crea automáticamente algunas para sus servicios (Cloud
  Build, Compute). También necesitan permisos para trabajar por ti.

### 3.2 · Un ROL es un paquete de permisos

Ejemplos: `roles/storage.admin` (gestionar buckets), `roles/run.admin` (gestionar Cloud Run). En vez
de dar permisos sueltos, das roles. **Principio de mínimo privilegio:** dar solo lo necesario.

### 3.3 · Las identidades en juego en este taller (y por qué cada permiso)

**a) La SA de runtime — con la que CORREN nuestros servicios.** La creamos nosotros:

```bash
gcloud iam service-accounts create taller-vision-sa --display-name="Taller Vision runtime"
# Permisos que necesita para hacer su trabajo:
gcloud projects add-iam-policy-binding PROJECT \
  --member="serviceAccount:taller-vision-sa@PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.admin"                       # leer/escribir el bucket (imágenes y modelo)
gcloud projects add-iam-policy-binding PROJECT \
  --member="serviceAccount:taller-vision-sa@PROJECT.iam.gserviceaccount.com" \
  --role="roles/serviceusage.serviceUsageConsumer"   # usar APIs (Vision) facturando a este proyecto
```

> **Idea clave para la sala:** cuando desplegamos el job/servicio con `--service-account
> taller-vision-sa@...`, el código **se autentica solo** como esa SA. No metemos contraseñas ni keys
> en el código: la identidad se la da Cloud Run. Si la SA tiene acceso al bucket, el código tiene
> acceso al bucket. Esto es **lo correcto** (vs. meter una key en el repo, que es el antipatrón).

**b) Las SAs de Cloud Build y Compute — las que CONSTRUYEN y DESPLIEGAN.** Ya existen (las crea
Google). Cuando hacemos `gcloud run deploy --source .`, por detrás:
1. **Cloud Build** construye la imagen del contenedor,
2. la sube a **Artifact Registry**,
3. la **despliega** en Cloud Run.

Para hacer eso necesitan roles. Por eso les damos:

```bash
# a la SA de Cloud Build y a la de Compute:
roles/run.admin                 # crear/actualizar servicios y jobs de Cloud Run
roles/artifactregistry.admin    # guardar la imagen del contenedor
roles/storage.admin             # subir el código fuente al bucket de build
roles/logging.logWriter         # escribir logs del build
roles/cloudbuild.builds.builder # ejecutar builds
roles/iam.serviceAccountUser    # *** ver abajo ***
```

> **¿Cloud Build SA o Compute SA?** Desde 2024, `gcloud run deploy --source` corre el build, por
> defecto, como la **Compute Engine default SA** (`NUMERO-compute@developer...`), no como la antigua
> `NUMERO@cloudbuild...`. Como no sabes la antigüedad del proyecto del asistente, **da los roles a las
> dos** y te ahorras el fallo. El síntoma despista mucho: el error habla de
> `storage.objects.get denied` (no puede leer el código fuente que se sube al bucket de staging del
> build), no de "build" — y uno se vuelve loco buscando por el lado equivocado. **Es el error #2.**

**c) El permiso que más se olvida — "actuar como" (`serviceAccountUser`).**

```bash
gcloud iam service-accounts add-iam-policy-binding taller-vision-sa@PROJECT.iam.gserviceaccount.com \
  --member="serviceAccount:CLOUDBUILD_SA" --role="roles/iam.serviceAccountUser"
```

Cuando desplegamos un servicio que **correrá como** `taller-vision-sa`, quien hace el despliegue
(Cloud Build/Compute) tiene que tener permiso para **"actuar como" esa SA**. Es como decir: "te
autorizo a poner en marcha algo en mi nombre". Si falta este permiso, el deploy falla con un error de
*iam.serviceAccounts.actAs*. **Este es el error de permisos #1 con Cloud Run + Cloud Build.**

### 3.4 · Tabla resumen (para una slide)

| Identidad | Es… | Para qué la usamos | Roles que le damos |
|---|---|---|---|
| `tu-usuario@example.com` | usuario | crear el proyecto, lanzar todo | (es Owner de la org) |
| `taller-vision-sa` | SA nuestra | **correr** job y servicio | `storage.admin`, `serviceUsageConsumer` |
| Cloud Build SA | SA de Google | **construir** la imagen | `run.admin`, `artifactregistry.admin`, `storage.admin`, `logging.logWriter`, `cloudbuild.builds.builder`, `serviceAccountUser` |
| Compute SA | SA de Google | runtime/deploy por defecto | (los mismos roles de despliegue) |

### 3.5 · Detalle real que vivimos (sirve de ejemplo)

- Al dar el primer rol a la SA **recién creada**, falló con *"service account does not exist"*: IAM
  tarda unos segundos en **propagar** una identidad nueva. Solución: reintentar. → Buen ejemplo de
  que IAM es **eventualmente consistente**.

---

## 4 · El bucket

```bash
gcloud storage buckets create gs://tu-proyecto-gcp-imagenes \
  --location=europe-west4 --uniform-bucket-level-access
```

- `--uniform-bucket-level-access`: los permisos se gestionan **solo por IAM** (no ACLs por objeto).
  Más simple y más seguro.
- Aquí viven dos cosas: las **imágenes** (`demo/`) y el **modelo entrenado** (`models/flores102/`).
- Región `europe-west4`: es una de las que tienen **GPU NVIDIA L4** para Cloud Run (el job entrena en GPU).

---

## 5 · Cloud Run: un JOB para entrenar y un SERVICE para servir

Cloud Run ejecuta contenedores. Tiene dos formas:
- **Job**: corre hasta **terminar** una tarea. Perfecto para **entrenar** (`taller-entrenar-flores`).
  Lo desplegamos **con GPU** (`--gpu 1 --gpu-type nvidia-l4`): entrena en GPU y al acabar **muere**
  (no cuesta nada en reposo).
- **Service**: responde a **HTTP** siempre. Perfecto para **servir** la inferencia
  (`taller-inferencia-flores`). Va **con GPU** (`--gpu 1 --gpu-type nvidia-l4`) y con
  `--min-instances 0` **se apaga solo** cuando nadie lo usa (la GPU deja de cobrar en reposo).

Ambos se despliegan con `--service-account taller-vision-sa@...` (corren como esa identidad) y
reciben su configuración por **variables de entorno** (`BUCKET`, `MODEL_GCS`, `EPOCHS`):

```bash
gcloud run jobs deploy taller-entrenar-flores --source . \
  --service-account taller-vision-sa@PROJECT.iam.gserviceaccount.com \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi \
  --set-env-vars BUCKET=...,MODEL_DIR=models/flores102,EPOCHS=40,IMG_SIZE=180

gcloud run deploy taller-inferencia-flores --source . \
  --service-account taller-vision-sa@PROJECT.iam.gserviceaccount.com \
  --min-instances 0 --set-env-vars MODEL_GCS=gs://.../models/flores102
```

- **Variables de entorno** = el código no "descubre" nada en caliente; lo lee al arrancar → más
  rápido y reproducible.
- **`--min-instances 1`** en el servicio: una instancia siempre caliente → **sin cold start** ni
  esperas de carga del modelo en la demo (el modelo se pre-carga al arrancar).

### 5.1 · Detalle real: la org bloqueó el acceso público (gran ejemplo de IAM)

Desplegamos el servicio con `--allow-unauthenticated` (queríamos un endpoint público), pero el deploy
avisó: *"Setting IAM policy failed"*. ¿Por qué? La organización `example.com` tiene una **org policy**
(`constraints/iam.allowedPolicyMemberDomains`, *domain restricted sharing*) que **prohíbe dar permiso
a `allUsers`**. Es una protección corporativa para que nadie publique recursos al mundo por error.

Resultado: el servicio quedó **privado**, y se llama **autenticado** con un id-token:

```python
tok = subprocess.run(["gcloud","auth","print-identity-token"], capture_output=True, text=True).stdout.strip()
requests.post(f"{URL}/predict", headers={"Authorization": f"Bearer {tok}"}, json={...})
```

> Mensaje para la sala: las **org policies** mandan por encima de tus permisos de proyecto. Y un
> servicio **privado + llamada autenticada** es, de hecho, **mejor práctica** que un endpoint público.
> El cuaderno ya lo hace por ti (la función `clasificar()` añade el token).

---

## 6 · Vision API y el "quota project" (otro ejemplo didáctico)

Al probar Vision con nuestro usuario, salió `403 ... requires a quota project`. La causa: Google no
sabía a **qué proyecto** cobrarle la cuota. Se arregla indicando el proyecto en una cabecera:

```bash
curl -H "X-Goog-User-Project: tu-proyecto-gcp" ...   # o, en código, set_quota_project / ADC
```

> Mensaje: cuando usas credenciales de **usuario** (no de SA), conviene fijar el **quota project**.
> Con la **SA de runtime** esto no pasa, porque su identidad ya está ligada al proyecto.

---

## 7 · Limpieza (para no dejar gasto)

```bash
gcloud run services delete taller-inferencia-flores --region europe-southwest1   # quita min-instances
gcloud run jobs delete taller-entrenar-flores --region europe-southwest1
# o lo más limpio de todo:
gcloud projects delete tu-proyecto-gcp
```

---

## Resumen en una frase para cerrar

> Montar visión en Google Cloud es **3 cosas**: un **proyecto** con sus **APIs activas**, **IAM** bien
> puesto (identidades que corren como SAs con el mínimo privilegio), y tus **servicios** (Storage,
> Vision, Cloud Run) hablando entre ellos. IAM es el 80% de los problemas y el 80% del aprendizaje.
