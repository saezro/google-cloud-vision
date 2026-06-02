"""Cloud Run JOB — entrena una CNN DESDE CERO (en GPU) y guarda el modelo en GCS.

No corre en Colab: se ejecuta en la nube como un *Cloud Run job* con GPU
(`--gpu 1 --gpu-type nvidia-l4`). El cuaderno solo lo lanza. TensorFlow detecta
la GPU automáticamente.

Dataset: **Oxford Flowers 102** (102 clases) vía `tensorflow_datasets`. Como la
partición `train` oficial es pequeña (pensada para fine-tuning), para entrenar
una CNN desde cero combinamos `train+test` y evaluamos en `validation`.

Lee config de variables de entorno (las pone el deploy/execute):
  BUCKET      bucket de GCS donde se guarda el modelo (obligatorio)
  MODEL_DIR   prefijo del modelo dentro del bucket   (def: models/flores102)
  EPOCHS      épocas de entrenamiento                (def: 40)
  IMG_SIZE    lado de la imagen cuadrada             (def: 180)
  BATCH       tamaño de lote                         (def: 64)

Salida en gs://BUCKET/MODEL_DIR/ :
  saved_model/   (SavedModel servible)
  metrics.json   (accuracy top-1/top-5, clases, historial, por clase…)
"""
import json
import os

import tensorflow as tf

SEED = 42
DATASET = "oxford_flowers102"


def construir_modelo(n_clases: int, img: int = 180) -> tf.keras.Model:
    """CNN desde cero, más profunda para aprovechar la GPU. Importable para enseñarla.

    Bloques:
      Rescaling + aumento de datos     normaliza y genera variaciones al vuelo
      4 × (Conv2D + BatchNorm + Pool)  extraen patrones (bordes→texturas→formas) y reducen tamaño
      GlobalAveragePooling             resume cada mapa de características en un número
      Dense + Dropout                  clasificador, con dropout para no sobreajustar
      Dense softmax                    una probabilidad por clase
    """
    from tensorflow.keras import layers, models
    return models.Sequential([
        layers.Input(shape=(img, img, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding="same", activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation="softmax"),
    ])


def preparar(img: int, batch: int):
    """Carga Flores-102 de tfds. Entrena con train+test (~7k), evalúa en validation."""
    import tensorflow_datasets as tfds  # import perezoso: el cuaderno importa construir_modelo sin tfds
    (ds_tr, ds_va), info = tfds.load(
        DATASET, split=["train+test", "validation"],
        as_supervised=True, with_info=True)
    clases = list(info.features["label"].names)

    def prep(x, y):
        return tf.image.resize(x, (img, img)), y

    A = tf.data.AUTOTUNE
    ds_tr = ds_tr.map(prep, A).shuffle(2048, seed=SEED).batch(batch).prefetch(A)
    ds_va = ds_va.map(prep, A).batch(batch).prefetch(A)
    return ds_tr, ds_va, clases


def main():
    BUCKET = os.environ["BUCKET"]
    MODEL_DIR = os.getenv("MODEL_DIR", "models/flores102").strip("/")
    EPOCHS = int(os.getenv("EPOCHS", "40"))
    IMG = int(os.getenv("IMG_SIZE", "180"))
    BATCH = int(os.getenv("BATCH", "64"))

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visibles: {gpus or 'NINGUNA (CPU)'}", flush=True)

    ds_tr, ds_va, clases = preparar(IMG, BATCH)
    print(f"Dataset {DATASET}: {len(clases)} clases", flush=True)

    model = construir_modelo(len(clases), IMG)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")])
    model.summary()

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS)
    val_acc = float(hist.history["val_accuracy"][-1])
    val_top5 = float(hist.history["val_top5"][-1])
    print(f"val_accuracy={val_acc:.3f}  val_top5={val_top5:.3f}", flush=True)

    # --- Evaluación por clase + matriz de confusión (sobre validation) ----------
    import numpy as np
    y_true, y_pred = [], []
    for bx, by in ds_va:
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
    model.export(destino_modelo)
    print(f"Modelo guardado en {destino_modelo}", flush=True)

    metrics = {
        "val_accuracy": round(val_acc, 4),
        "val_top5": round(val_top5, 4),
        "classes": clases,
        "img_size": IMG,
        "epochs": EPOCHS,
        "dataset": DATASET,
        "history": {k: [round(float(x), 4) for x in v] for k, v in hist.history.items()},
        "accuracy_por_clase": por_clase,
        "matriz_confusion": confusion,
    }
    with tf.io.gfile.GFile(f"gs://{BUCKET}/{MODEL_DIR}/metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print("metrics.json escrito. ENTRENAMIENTO COMPLETO.", flush=True)


if __name__ == "__main__":
    main()
