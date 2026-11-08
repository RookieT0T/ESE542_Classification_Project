"""Shared data loading + the train/validation split used by every script.

The official test set is NOT loaded anywhere in this file (or in Phase 1 at all).
"""
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

# Reuse the loader that ships with the Fashion-MNIST repo.
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "fashion-mnist"))
from utils.mnist_reader import load_mnist  # noqa: E402

SEED = 0  # one fixed seed, so everyone gets the identical split


def load_train_val():
    """Return (X_train, y_train, X_val, y_val).

    60,000 official training images -> 50,000 train + 10,000 validation.
    The split is stratified, i.e. every class keeps the same share in both parts.
    """
    X, y = load_mnist(str(HERE / "fashion-mnist" / "data" / "fashion"), kind="train")
    X = X.astype("float32") / 255.0  # pixels 0..255 -> 0..1 (uses nothing learned from data)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=10_000, stratify=y, random_state=SEED
    )
    return X_train, y_train, X_val, y_val
