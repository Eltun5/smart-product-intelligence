"""
Model architectures and loading helpers for the
Smart Product Intelligence project.
"""

import os
import json
from typing import Tuple, Dict


IMG_SIZE = 128
DEFAULT_MODEL_DIR = "/content/drive/MyDrive/smart-product-intelligence"


def build_tabular_mlp(input_dim: int = 8, num_classes: int = 3):
    """Build the M1 MLP (8 -> 64 -> BN -> Drop -> 32 -> BN -> Drop -> 16 -> num_classes)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_cnn_from_scratch(num_classes: int = 6):
    """Small from-scratch CNN used as the baseline in M2."""
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_transfer_cnn(num_classes: int = 6, fine_tune: bool = False):
    """MobileNetV2-based transfer-learning model for M2."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2

    backbone = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                           include_top=False, weights="imagenet")
    backbone.trainable = fine_tune

    model = tf.keras.Sequential([
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def load_m1(model_dir: str = DEFAULT_MODEL_DIR):
    """Load M1 MLP and feature scaler."""
    import tensorflow as tf
    import joblib

    model = tf.keras.models.load_model(os.path.join(model_dir, "m1_tabular_mlp.h5"))
    scaler_path = os.path.join(model_dir, "m1_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


def load_m2(model_dir: str = DEFAULT_MODEL_DIR) -> Tuple:
    """Load M2 MobileNetV2 model and index -> category map."""
    import tensorflow as tf

    model = tf.keras.models.load_model(os.path.join(model_dir, "m2_cnn_transfer.h5"))
    with open(os.path.join(model_dir, "m2_categories.json")) as f:
        categories = {int(k): v for k, v in json.load(f).items()}
    return model, categories


def load_m4(model_dir: str = DEFAULT_MODEL_DIR):
    """Load fine-tuned DistilBERT classifier."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    m4_path = os.path.join(model_dir, "m4_distilbert")
    tokenizer = AutoTokenizer.from_pretrained(m4_path)
    model = AutoModelForSequenceClassification.from_pretrained(m4_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer, device


def load_embeddings(model_dir: str = DEFAULT_MODEL_DIR):
    """Load cached product embeddings and parallel index."""
    import numpy as np
    import pandas as pd

    embeddings = np.load(os.path.join(model_dir, "m3_product_embeddings.npy"))
    index = pd.read_csv(os.path.join(model_dir, "m3_product_index.csv"))
    return embeddings, index
