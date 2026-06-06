"""
Evaluation and visualization helpers for the
Smart Product Intelligence project.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def classification_summary(y_true, y_pred, average: str = "macro") -> Dict:
    """Return accuracy, macro-F1, precision and recall in one dict."""
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


def plot_confusion_matrix(y_true, y_pred, class_names: List[str],
                          title: str = "Confusion Matrix", cmap: str = "Blues",
                          normalize: bool = True, save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6)):
    """Plot a normalized confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"label": "Proportion" if normalize else "Count"})
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


def predict_rating_band(model, scaler, price: float, rating_number: int,
                        title: str, description: str = "") -> Tuple[str, float]:
    """Run a rating-band prediction with the M1 MLP."""
    pwm = 1 if (price is None or price == 0) else 0
    pf = float(price) if price else 20.0
    features = np.array([[pf, pwm, rating_number or 0, 0, 0, 100,
                          len(str(title or "")), len(str(description or ""))]],
                        dtype=np.float32)
    if scaler is not None:
        features = scaler.transform(features)
    pred = model.predict(features, verbose=0)
    bands = ["low", "medium", "high"]
    idx = int(np.argmax(pred))
    return bands[idx], float(pred[0][idx]) * 100


def predict_image_category(model, categories: Dict[int, str],
                           pil_image, img_size: int = 128) -> Tuple[str, float]:
    """Run subcategory prediction with the M2 MobileNetV2 model."""
    img = pil_image.convert("RGB").resize((img_size, img_size))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    pred = model.predict(arr, verbose=0)
    idx = int(np.argmax(pred))
    return categories[idx], float(pred[0][idx]) * 100


def find_similar_products(query: str, embedder, product_embeddings: np.ndarray,
                          product_index: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """Semantic search over cached product embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_vec, product_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {"title": product_index.iloc[i]["title"][:80],
         "asin": product_index.iloc[i]["parent_asin"],
         "similarity": round(float(sims[i]), 3)}
        for i in top_idx
    ]


def save_metrics(metrics: Dict, path: str) -> None:
    """Save a metrics dictionary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> Dict:
    """Load a metrics JSON dictionary."""
    with open(path) as f:
        return json.load(f)
