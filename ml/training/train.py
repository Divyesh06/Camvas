import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)

CLASSES = ['square', 'circle', 'triangle', 'line', 'arrow']
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 28
MODEL_PATH = "ml/models/ShapeDetection.keras"


def build_model():
    """Small CNN with built-in augmentation.

    Augmentation layers are active only when `training=True` (i.e. during
    model.fit). During inference and ONNX export they pass through as
    identity, so they don't affect the exported model.
    """
    model = Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.RandomRotation(0.15, fill_mode='constant', fill_value=0.0),
        layers.RandomZoom(0.15, fill_mode='constant', fill_value=0.0),
        layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    return model


def load_from_pickle():
    with open("ml/models/features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("ml/models/labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels


if __name__ == "__main__":
    features, labels = load_from_pickle()
    labels = labels.flatten().astype(np.int32)
    features, labels = shuffle(features, labels, random_state=0)

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, random_state=0, test_size=0.1, stratify=labels
    )

    # Normalize exactly once, here. Pickle stores uint8 [0, 255].
    train_x = train_x.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255.0
    test_x = test_x.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255.0

    print(f"Train: {train_x.shape}, Test: {test_x.shape}")
    print(f"Train class dist: {np.bincount(train_y)}")
    print(f"Test class dist:  {np.bincount(test_y)}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        TensorBoard(log_dir="ShapeDetection"),
    ]

    model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    loss, acc = model.evaluate(test_x, test_y, verbose=0)
    print(f"Final test accuracy: {acc:.4f} (loss {loss:.4f})")
