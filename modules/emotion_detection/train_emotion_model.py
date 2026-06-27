import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Flatten, Dense, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import os

# Emotions the model will classify
EMOTIONS     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE     = 48
BATCH_SIZE   = 64
EPOCHS       = 50
MODEL_PATH   = Path("modules/emotion_detection/models/emotion_model.h5")


def build_model() -> Sequential:
    """
    CNN architecture designed for 48x48 grayscale face images.
    3 conv blocks + dropout for regularisation.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # Classifier head
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(EMOTIONS), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train(data_dir: str = "data/raw/fer2013"):
    """
    Train on FER-2013 dataset.

    Download dataset from:
    https://www.kaggle.com/datasets/msambare/fer2013
    Extract to: data/raw/fer2013/
    Expected structure:
        data/raw/fer2013/train/<emotion_name>/*.jpg
        data/raw/fer2013/test/<emotion_name>/*.jpg
    """
    train_dir = Path(data_dir) / "train"
    test_dir  = Path(data_dir) / "test"

    if not train_dir.exists():
        print("Dataset not found. Please download FER-2013 from Kaggle.")
        print("Extract to: data/raw/fer2013/")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        return

    # Augmentation for training — helps model generalise better
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_model()
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Stop early if val_loss stops improving for 10 epochs
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        # Reduce LR when stuck
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        # Save best model automatically
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True),
    ]

    print("\nTraining started...")
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    final_acc = max(history.history["val_accuracy"])
    print(f"\nBest Validation Accuracy: {final_acc:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()