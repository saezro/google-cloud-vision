# Entrenar y servir tu propio modelo en Google Cloud

Un taller práctico, paso a paso, para llevar un modelo de imágenes **de un dataset a producción, todo
en tu nube**: lo entrenas en un **job de Cloud Run** y lo sirves en un **service de Cloud Run**. Y de
paso, ves cómo servir un modelo **pre-entrenado** sin entrenar nada. Colab solo manda; el cómputo va
en Google Cloud.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saezro/google-cloud-vision/blob/main/google-cloud-vision.ipynb)

---

## La idea

El hilo es **entrenar un modelo desde cero y ponerlo a servir inferencias**, y ver que la infra de
servir es la misma sea cual sea el modelo:

- **Tu CNN** — el grueso: con tus clases, la entrenas en un **job** de Cloud Run y la sirves en un
  **service** de Cloud Run.
- **Un modelo pre-entrenado** — para cerrar: descargas MobileNet (ImageNet) y lo sirves **en el mismo
  service**, sin entrenar nada. Mismo endpoint, otro modelo.

Todo corre en **tu** proyecto. Nada externo (no es la Vision API, que correría en servidores de
Google): aquí el modelo es un fichero en tu bucket que sirve tu Cloud Run.

## Cómo está montado

```
   Colab (el mando)  ── órdenes por API ──►   Google Cloud
        |                                      Cloud Storage     imágenes + modelos
   eliges proyecto, lanzas, miras             Cloud Run JOB      entrena la CNN
                                              Cloud Run SERVICE  sirve cualquier modelo
```

Colab no es Google Cloud: es solo el cliente que da las órdenes. No hay máquinas (VM) ni Compute
Engine, y si cierras Colab lo que está en Cloud Run sigue funcionando.

**Job vs Service** es la distinción clave: un **job** arranca, hace una tarea y muere (entrenar); un
**service** se queda escuchando peticiones (servir). Mismo Cloud Run, dos modos.

## Cómo se usa

1. Abre el cuaderno en Colab con el botón de arriba.
2. Si quieres quedártelo, `Archivo → Guardar una copia en Drive`.
3. Dale al play de arriba a abajo. La primera celda se baja sola el código y las imágenes.
4. Te conectas con tu cuenta y eliges tu proyecto en el desplegable. No hay nada que configurar a mano.

Puedes trastear sin miedo: lo que edites en Colab es tuyo, no toca el repo ni a nadie más.

## Qué va haciendo

1. Activa las APIs y monta el **IAM** (una service account con la que corre todo).
2. Crea un **bucket** y sube unas imágenes.
3. Entrena una CNN de flores **desde cero** en un **job** de Cloud Run y guarda el modelo en el bucket.
4. Enseña las gráficas de cómo ha aprendido.
5. Despliega un **service** de Cloud Run y clasifica fotos nuevas con tu modelo.
6. Descarga un modelo **pre-entrenado** (MobileNet/ImageNet) y lo sirve en el **mismo** service.

## Qué hay en el repo

| Ruta | Qué es |
|---|---|
| `google-cloud-vision.ipynb` | El cuaderno que lo orquesta todo. |
| `cloud/entrenamiento/` | El **job** de Cloud Run que entrena la CNN. |
| `cloud/inferencia/` | El **service** de Cloud Run que sirve los modelos (agnóstico al modelo). |
| `imagenes/` | Imágenes de prueba. |
| `docs/SETUP-Y-IAM-EXPLICADO.md` | Cómo se monta la infra, con foco en IAM. |
| `tools/build_notebook.py` | Genera el cuaderno (mejor no tocar el `.ipynb` a mano). |

## Lo que necesitas

- Un proyecto de Google Cloud con **billing** (se crea desde la web, un par de clics).
- Permisos para crear cuentas de servicio. El IAM lo hace el cuaderno por ti.

## Cuánto cuesta

Casi nada: Storage son céntimos y Cloud Run entra en la capa gratis para una demo. Con el service a
`--min-instances 0` no pagas en reposo, y el job solo cuesta mientras entrena. Todo es CPU, **sin GPU**.

## Para limpiar

Cuando acabes, esto se lo lleva todo por delante:

```bash
gcloud projects delete TU_PROYECTO
```

## Abrir en Colab

https://colab.research.google.com/github/saezro/google-cloud-vision/blob/main/google-cloud-vision.ipynb

---

Hecho por [**@saezro**](https://github.com/saezro)
