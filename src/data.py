"""
Data loading, splitting, and feature engineering for the
Smart Product Intelligence project.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


DEFAULT_DATA_DIR = "/content/drive/MyDrive/smart-product-intelligence/data"


def load_products(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the cached products CSV."""
    return pd.read_csv(os.path.join(data_dir, "products.csv"))


def load_reviews(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the cached reviews CSV."""
    return pd.read_csv(os.path.join(data_dir, "reviews.csv"))


def load_image_index(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the image index (parent_asin -> local image_path)."""
    return pd.read_csv(os.path.join(data_dir, "image_index.csv"))


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return rows for a given split (train / val / test)."""
    return df[df["split"] == split].reset_index(drop=True)


def verify_no_leakage(products: pd.DataFrame) -> dict:
    """Confirm that no product appears in more than one split."""
    train_ids = set(products[products["split"] == "train"]["parent_asin"])
    val_ids = set(products[products["split"] == "val"]["parent_asin"])
    test_ids = set(products[products["split"] == "test"]["parent_asin"])
    return {
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "train_val_intersection": len(train_ids & val_ids),
        "train_test_intersection": len(train_ids & test_ids),
        "val_test_intersection": len(val_ids & test_ids),
    }


M2_SUBCATEGORIES = [
    "Games & Accessories",
    "Toy Figures & Playsets",
    "Party Supplies",
    "Puzzles",
    "Dolls & Accessories",
    "Stuffed Animals & Plush Toys",
]


def find_subcategory(cat_str: str) -> Optional[str]:
    """Match a category string against the six target classes."""
    if not isinstance(cat_str, str):
        return None
    for sub in M2_SUBCATEGORIES:
        if sub in cat_str:
            return sub
    return None


def make_tabular_features(products_df: pd.DataFrame,
                          reviews_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Build the 8-feature matrix used by the M1 MLP."""
    feats = pd.DataFrame(index=products_df.index)
    feats["price_filled"] = products_df["price_clean"].fillna(20.0)
    feats["price_was_missing"] = products_df["price_clean"].isna().astype(int)
    feats["rating_number"] = products_df["rating_number"].fillna(0)

    if reviews_df is not None:
        agg = (reviews_df.groupby("parent_asin")
               .agg(n_reviews=("rating", "count"),
                    avg_helpful=("helpful_vote", "mean"),
                    avg_review_len=("text_length", "mean"))
               .reset_index())
        merged = products_df.merge(agg, on="parent_asin", how="left")
        feats["n_reviews_in_sample"] = merged["n_reviews"].fillna(0).values
        feats["avg_helpful"] = merged["avg_helpful"].fillna(0).values
        feats["avg_review_len"] = merged["avg_review_len"].fillna(100).values
    else:
        feats["n_reviews_in_sample"] = 0
        feats["avg_helpful"] = 0
        feats["avg_review_len"] = 100

    feats["title_len"] = products_df["title"].astype(str).str.len()
    feats["desc_len"] = products_df["description_text"].astype(str).str.len()
    return feats.values.astype(np.float32)


def make_rating_band_target(products_df: pd.DataFrame) -> np.ndarray:
    """Bin average_rating into 3 classes (low / medium / high)."""
    return pd.cut(
        products_df["average_rating"].fillna(0),
        bins=[-0.1, 3.5, 4.3, 5.1],
        labels=[0, 1, 2],
    ).astype(int).values


def make_sentiment_dataset(reviews_df: pd.DataFrame,
                           min_text_len: int = 10) -> pd.DataFrame:
    """Convert review ratings to binary sentiment (drop 3-star)."""
    df = reviews_df[reviews_df["rating"].isin([1, 2, 4, 5])].copy()
    df["sentiment"] = (df["rating"] >= 4).astype(int)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.len() > min_text_len]
    return df.reset_index(drop=True)
