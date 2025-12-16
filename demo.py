# =========================
# Chest X-ray Pneumonia CNN
# FINAL DEMO VERSION
# =========================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# Dataset path
# -------------------------
DATA_DIR = r"C:\Users\Nathan\Desktop\chest_xray\chest_xray"
train_dir = os.path.join(DATA_DIR, "train")
test_dir  = os.path.join(DATA_DIR, "test")

# -------------------------
# Settings
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
THRESHOLD = 0.75     # decision threshold
N_SHOW = 8           # number of images to show in demo
COLS = 4             # images per row

# -------------------------
# Load datasets
# -------------------------
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.15,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.15,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

test_raw = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# -------------------------
# Normalize images
# -------------------------
normalizer = keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (normalizer(x), y)).prefetch(tf.data.AUTOTUNE)

# -------------------------
# Build CNN (no warnings)
# -------------------------
model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),

    keras.layers.Conv2D(32, 7, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train model
# -------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# -------------------------
# DEMO OUTPUT (Fashion-MNIST style grid)
# -------------------------
ROWS = (N_SHOW + COLS - 1) // COLS

for images, labels in test_raw.take(1):
    images = images[:N_SHOW]

    probs = model.predict(images / 255.0, verbose=0).ravel()
    preds = (probs > THRESHOLD).astype(int)

    plt.figure(figsize=(12, 6))
    for i in range(N_SHOW):
        plt.subplot(ROWS, COLS, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

        pred_name = class_names[preds[i]]
        plt.title(f"{pred_name}\nP={probs[i]:.2f}", fontsize=10)

    plt.tight_layout()
    plt.show()
    break
