# Google Cloud Vision — de una foto a una decisión

Un proyecto para ver, paso a paso, cómo convertir una imagen en algo útil con Google Cloud. Dos
caminos: la **Vision API** (visión ya hecha, sin entrenar nada) y una **CNN tuya**, entrenada y
servida en **Cloud Run**. Todo corre en la nube; Colab solo manda.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saezro/google-cloud-vision/blob/main/google-cloud-vision.ipynb)

---

## La idea

Una foto no es un dato hasta que algo la convierte en una decisión. Aquí montamos ese "algo" de dos
formas y se ve cuándo conviene cada una:

- **Vision API** — etiquetas, objetos y texto (OCR) sin entrenar nada.
- **CNN propia** — cuando las clases son tuyas, entrenas tu modelo y lo sirves como una API.

## Cómo está montado

```
   Colab (el mando)  ── órdenes por API ──►   Google Cloud
        |                                      Cloud Storage    imágenes + modelo
   eliges proyecto, lanzas, miras             Vision API       visión gestionada
                                              Cloud Run JOB    entrena la CNN
                                              Cloud Run SERVICE sirve la CNN
```

Colab no es Google Cloud: es solo el cliente que da las órdenes. No hay máquinas (VM) ni Compute
Engine, y si cierras Colab lo que está en Cloud Run sigue funcionando.

## Cómo se usa

1. Abre el cuaderno en Colab con el botón de arriba.
2. Si quieres quedártelo, `Archivo → Guardar una copia en Drive`.
3. Dale al play de arriba a abajo. La primera celda se baja sola el código y las imágenes.
4. Eliges tu proyecto en el desplegable y listo.

Puedes trastear sin miedo: lo que edites en Colab es tuyo, no toca el repo ni a nadie más.

## Qué va haciendo

1. Crea un bucket y sube unas imágenes.
2. Pasa una foto por la Vision API y saca un veredicto.
3. Entrena una CNN de flores en un job de Cloud Run y guarda el modelo en el bucket.
4. Enseña las gráficas de cómo ha aprendido.
5. Sirve el modelo en Cloud Run y clasifica fotos nuevas.

## Qué hay en el repo

| Ruta | Qué es |
|---|---|
| `google-cloud-vision.ipynb` | El cuaderno que lo orquesta todo. |
| `cloud/entrenamiento/` | El job de Cloud Run que entrena la CNN. |
| `cloud/inferencia/` | El servicio de Cloud Run que la sirve. |
| `imagenes/` | Imágenes de prueba. |
| `docs/SETUP-Y-IAM-EXPLICADO.md` | Cómo se monta la infra, con foco en IAM. |
| `tools/build_notebook.py` | Genera el cuaderno (mejor no tocar el `.ipynb` a mano). |
| `demo/` | La parte de Vision/Storage en Python suelto, por si la quieres en local. |

## Lo que necesitas

- Un proyecto de Google Cloud con **billing** (se crea desde la web, un par de clics).
- Permisos para crear cuentas de servicio. El IAM lo hace el cuaderno por ti.

## Cuánto cuesta

Casi nada: la Vision API regala 1.000 usos al mes, Storage son céntimos y Cloud Run entra en la capa
gratis para una demo. Todo es CPU, **sin GPU**.

## Para limpiar

Cuando acabes, esto se lo lleva todo por delante:

```bash
gcloud projects delete TU_PROYECTO
```

## Abrir en Colab

https://colab.research.google.com/github/saezro/google-cloud-vision/blob/main/google-cloud-vision.ipynb

---

Hecho por [**@saezro**](https://github.com/saezro)
