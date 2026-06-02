"""Cloud Run JOB — entrena una CNN de flores y guarda el modelo en GCS.

Esto NO corre en Colab: se ejecuta en la nube como un *Cloud Run job*. El
notebook del taller solo lo lanza ("darle al play"). Todo el cómputo (descargar
dataset, entrenar, guardar modelo) ocurre aquí, en Cloud Run.

Lee config de variables de entorno (las pone el deploy/execute):
  BUCKET        bucket de GCS donde se guarda el modelo (obligatorio)
  MODEL_DIR     prefijo del modelo dentro del bucket  (def: models/flores)
  EPOCHS        épocas de entrenamiento               (def: 8)
  IMG_SIZE      lado de la imagen cuadrada             (def: 128)

Salida en gs://BUCKET/MODEL_DIR/ :
  saved_model/   (SavedModel de Keras)
  metrics.json   (accuracy, clases, parámetros)
"""
import json
import os
import tarfile
import urllib.request
from pathlib import Path

import tensorflow as tf

SEED = 42

# Dataset público de flores que Google sirve... desde Cloud Storage.
DATASET_URL = ("https://storage.googleapis.com/download.tensorflow.org/"
               "example_images/flower_photos.tgz")


def preparar_dataset() -> Path:
    """Descarga y extrae flower_photos. Devuelve el dir con subcarpetas/clase."""
    destino = Path("/tmp/flores")
    data_dir = destino / "flower_photos"
    if data_dir.is_dir():
        return data_dir
    destino.mkdir(parents=True, exist_ok=True)
    tgz = destino / "flower_photos.tgz"
    print(f"Descargando dataset {DATASET_URL} ...", flush=True)
    urllib.request.urlretrieve(DATASET_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(destino)
    print("Dataset listo.", flush=True)
    return data_dir


def construir_modelo(n_clases: int, img: int = 128) -> tf.keras.Model:
    """La CNN del taller: pequeña pero honesta. Importable desde el cuaderno para enseñarla.

    Capas:
      Rescaling          normaliza píxeles 0-255 -> 0-1
      RandomFlip/Rotation aumento de datos (solo activo en entrenamiento)
      Conv2D + MaxPool x3 extraen patrones (bordes -> texturas -> formas) y reducen tamaño
      GlobalAveragePooling resume cada mapa de características en un número
      Dense + Dropout      clasificador, con dropout para no sobreajustar
      Dense softmax        una probabilidad por clase
    """
    from tensorflow.keras import layers, models
    return models.Sequential([
        layers.Input(shape=(img, img, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation="softmax"),
    ])


def main():
    BUCKET = os.environ["BUCKET"]
    MODEL_DIR = os.getenv("MODEL_DIR", "models/flores").strip("/")
    EPOCHS = int(os.getenv("EPOCHS", "8"))
    IMG = int(os.getenv("IMG_SIZE", "128"))
    BATCH = int(os.getenv("BATCH", "32"))

    data_dir = preparar_dataset()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=(IMG, IMG), batch_size=BATCH)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=(IMG, IMG), batch_size=BATCH)
    clases = train_ds.class_names
    print("Clases:", clases, flush=True)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model = construir_modelo(len(clases), IMG)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    val_acc = float(hist.history["val_accuracy"][-1])
    print(f"val_accuracy final = {val_acc:.3f}", flush=True)

    # --- Estadísticas para el cuaderno (curvas + matriz de confusión) ----------
    import numpy as np
    y_true, y_pred = [], []
    for bx, by in val_ds:
        p = model.predict(bx, verbose=0)
        y_pred.extend(np.argmax(p, axis=1).tolist())
        y_true.extend(by.numpy().tolist())
    n = len(clases)
    confusion = [[0] * n for _ in range(n)]
    for t, pr in zip(y_true, y_pred):
        confusion[t][pr] += 1
    por_clase = {clases[i]: round(confusion[i][i] / max(sum(confusion[i]), 1), 3)
                 for i in range(n)}

    # Guardar SavedModel directamente en GCS (TF habla gs:// de forma nativa).
    destino_modelo = f"gs://{BUCKET}/{MODEL_DIR}/saved_model"
    model.export(destino_modelo)   # SavedModel servible
    print(f"Modelo guardado en {destino_modelo}", flush=True)

    metrics = {
        "val_accuracy": round(val_acc, 4),
        "classes": clases,
        "img_size": IMG,
        "epochs": EPOCHS,
        "dataset": "tf_flowers (flower_photos)",
        "n_train": int(train_ds.cardinality().numpy() * BATCH),
        "history": {k: [round(float(x), 4) for x in v] for k, v in hist.history.items()},
        "accuracy_por_clase": por_clase,
        "matriz_confusion": confusion,
    }
    with tf.io.gfile.GFile(f"gs://{BUCKET}/{MODEL_DIR}/metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print("metrics.json escrito. ENTRENAMIENTO COMPLETO.", flush=True)


if __name__ == "__main__":
    main()
