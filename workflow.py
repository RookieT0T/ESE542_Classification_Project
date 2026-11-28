"""Run profiles and durable output helpers shared by the experiment scripts."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import uuid

RATIOS = (0.2, 0.4, 0.6, 0.8, 1.0)
SEEDS = tuple(range(10))
ROOT = Path(__file__).resolve().parent


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true")
    return parser


def output(smoke):
    path = ROOT / "results" / ("smoke" if smoke else "final")
    path.mkdir(parents=True, exist_ok=True)
    return path


def grid(smoke):
    if smoke:
        return [(ratio, seed) for ratio in (0.4, 1.0) for seed in (0, 1)]
    return [(ratio, seed) for ratio in RATIOS for seed in SEEDS]


def data(smoke):
    from common import load_train_val

    X, y, X_val, y_val = load_train_val()
    # The original split is made first. Smoke only takes slices within each partition.
    if smoke:
        return X[:2000], y[:2000], X_val[:500], y_val[:500]
    return X, y, X_val, y_val


def atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}-{uuid.uuid4().hex}{path.suffix}")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    atomic(path, lambda tmp: tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def read_json(path):
    return json.loads(path.read_text())


def record_config(path, config):
    if path.exists() and read_json(path) != config:
        raise SystemExit(f"Configuration differs from {path}. Use the original environment or archive that run first.")
    save_json(path, config)


def save_csv(path, rows):
    if rows:
        def write(tmp):
            with tmp.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        atomic(path, write)


def host():
    return {"machine": platform.node(), "platform": platform.platform(), "cpu": platform.processor()}
