"""
SMART AGRI AI - Plant Disease CNN Training Pipeline
Builds and trains a Deep Learning Convolutional Neural Network (CNN) on leaf pathology imagery.
Authentic training pipeline: does not generate fake weights or fabricate accuracy.
"""

import os
import sys
import json
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DISEASE_DATA_DIR,
    DISEASE_MODEL_PATH,
    DISEASE_CLASSES_PATH,
    DISEASE_METRICS_PATH
)

def train(epochs=10, batch_size=32, img_size=(224, 224)):
    print("=" * 65)
    print("  SMART AGRI AI - PLANT DISEASE CNN TRAINING PIPELINE")
    print("=" * 65)

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("[Notice] TensorFlow is not installed in the active environment.")
        print("To enable Deep Learning / CNN training, install TensorFlow:")
        print("  pip install tensorflow")
        print("\nNote: The web application will remain fully functional, reporting that")
        print("the disease model is waiting for dataset upload and CNN training.")
        return

    # Check for image classes
    if not DISEASE_DATA_DIR.exists():
        print(f"[Error] Disease dataset directory not found at: {DISEASE_DATA_DIR}")
        return

    subdirs = [d for d in DISEASE_DATA_DIR.iterdir() if d.is_dir()]
    valid_classes = []
    for d in subdirs:
        # Check if contains image files
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        if len(imgs) > 0:
            valid_classes.append((d.name, len(imgs)))

    if len(valid_classes) < 2:
        print("\n[Notice] Insufficient image classes detected in 'data/disease_dataset/'.")
        print("Found:", valid_classes if valid_classes else "None")
        print("\nTo train the Deep Learning CNN model:")
        print("1. Place plant leaf images into class subdirectories:")
        print("   data/disease_dataset/")
        print("   ├── Tomato___Early_blight/ (with .jpg images)")
        print("   ├── Tomato___Bacterial_spot/")
        print("   └── Tomato___healthy/")
        print("2. Re-run this script:")
        print("   python backend/training/train_disease_model.py")
        print("\nThe web UI will accurately state: 'Plant Disease CNN model is not trained yet.'")
        return

    print(f"[*] Found {len(valid_classes)} leaf disease classes:")
    for cls_name, count in valid_classes:
        print(f"    - {cls_name}: {count} images")

    print(f"\n[*] Loading image datasets (image size: {img_size}, batch size: {batch_size})...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DISEASE_DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DISEASE_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n[*] Constructing Deep Convolutional Neural Network (CNN) Architecture...")
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print(f"\n[*] Commencing CNN training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[OK] Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")

    # Save model weights
    DISEASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(DISEASE_MODEL_PATH))
    print(f"[+] Saved model to: {DISEASE_MODEL_PATH}")

    # Save class indices
    with open(DISEASE_CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[+] Saved class index labels to: {DISEASE_CLASSES_PATH}")

    # Save genuine metrics for dashboard
    metrics_payload = {
        "status": "ready",
        "model_name": "Deep Convolutional Neural Network (CNN)",
        "framework": f"TensorFlow {tf.__version__}",
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "input_resolution": f"{img_size[0]}x{img_size[1]}x3",
        "num_classes": num_classes,
        "classes": class_names,
        "accuracy": round(float(val_acc) * 100, 2),
        "loss": round(float(val_loss), 4),
        "history": {
            "accuracy": [round(float(x) * 100, 2) for x in history.history.get("accuracy", [])],
            "val_accuracy": [round(float(x) * 100, 2) for x in history.history.get("val_accuracy", [])]
        }
    }

    with open(DISEASE_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {DISEASE_METRICS_PATH}")
    print("[OK] Disease model training completed successfully.\n")

if __name__ == "__main__":
    train()
