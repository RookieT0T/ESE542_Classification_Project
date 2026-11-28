"""Freeze fitted models and export aligned per-image predictions."""
import hashlib

import numpy as np
import torch

from workflow import arguments, atomic, data, grid, output, read_json, record_config, save_csv
from xgb_setup import xgb

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 5420


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(out, smoke):
    if not (out / "timing_summary.csv").is_file():
        raise SystemExit("Finish the fixed-round timing benchmark before freezing.")
    baseline = read_json(out / "logreg.json")
    expected_train, expected_val = (2000, 500) if smoke else (50000, 10000)
    if (
        baseline["K"] != 5 or baseline["n_train"] != expected_train or baseline["n_val"] != expected_val
        or any(len(baseline["cv"][str(c)]["folds"]) != 5 for c in baseline["grid"])
    ):
        raise ValueError("Incomplete or incompatible baseline")
    models = [{
        "id": "logreg", "path": "logreg.pt", "ratio": None, "seed": None,
        "prediction_rounds": None, "sha256": digest(out / "logreg.pt"),
        "record_sha256": digest(out / "logreg.json"),
    }]
    for ratio, seed in grid(smoke):
        name = f"ratio{ratio}_seed{seed}"
        record = out / "xgb" / f"{name}.json"
        run = read_json(record)
        metrics = run["metrics"]
        if (
            (metrics["ratio"], metrics["seed"]) != (ratio, seed)
            or (run["n_train"], run["n_val"]) != (expected_train, expected_val)
            or metrics["prediction_rounds"] != metrics["best_round"] + 1
            or not 1 <= metrics["prediction_rounds"] <= metrics["rounds_run"]
        ):
            raise ValueError(f"Incompatible run record: {record}")
        models.append({
            "id": name, "path": run["model_path"], "ratio": ratio, "seed": seed,
            "prediction_rounds": metrics["prediction_rounds"],
            "sha256": digest(out / run["model_path"]), "record_sha256": digest(record),
        })
    manifest = {
        "source": "validation" if smoke else "official_test",
        "n_images": expected_val, "class_order": list(range(10)), "models": models,
        "scoring": {"xgboost": xgb.__version__, "torch": str(torch.__version__)},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "reference": 1.0},
    }
    record_config(out / "frozen_models.json", manifest)
    return manifest


def load_model(out, entry):
    path = out / entry["path"]
    if entry["id"] == "logreg":
        model = torch.nn.Linear(784, 10)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model.eval()
    model = xgb.Booster(params={"device": "cpu", "nthread": 2})
    model.load_model(path)
    model.set_param({"device": "cpu", "nthread": 2})
    return model


def main():
    parser = arguments(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="Record model hashes and round limits without scoring.")
    mode.add_argument("--official-test", action="store_true", help="Score the frozen models on the official test set.")
    args = parser.parse_args()
    if args.smoke and args.official_test:
        parser.error("Smoke scoring uses validation only.")
    if not (args.smoke or args.freeze or args.official_test):
        parser.error("Choose --freeze, --official-test, or --smoke.")
    out = output(args.smoke)
    if args.official_test and not (out / "frozen_models.json").is_file():
        parser.error("Run --freeze before opening the official test set.")
    manifest = freeze(out, args.smoke)
    if args.freeze:
        print(f"Froze {len(manifest['models'])} models in {out}")
        return

    predictions_dir = out / f"{manifest['source']}_predictions"
    missing = [
        entry for entry in manifest["models"]
        if not (predictions_dir / f"{entry['id']}.npz").is_file()
    ]
    if missing:
        # Check checkpoint readability before opening any held-out images.
        for entry in missing:
            model = load_model(out, entry)
            if entry["id"] != "logreg" and model.num_boosted_rounds() < entry["prediction_rounds"]:
                raise ValueError(f"Checkpoint has too few rounds: {entry['id']}")
        if args.smoke:
            _, _, X, y = data(True)
        else:
            from common import HERE, load_mnist

            X, y = load_mnist(str(HERE / "fashion-mnist" / "data" / "fashion"), kind="t10k")
            X = X.astype("float32") / 255.0
        if len(y) != manifest["n_images"]:
            raise ValueError(f"Unexpected number of scoring images: {len(y)}")
        image_ids = np.arange(len(y))
        matrix = xgb.DMatrix(X, nthread=2)
        for entry in missing:
            model = load_model(out, entry)
            if entry["id"] == "logreg":
                with torch.inference_mode():
                    probabilities = model(torch.from_numpy(X)).softmax(dim=1).numpy()
            else:
                probabilities = model.predict(matrix, iteration_range=(0, entry["prediction_rounds"]))
            if (
                probabilities.shape != (len(y), 10) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1)
            ):
                raise ValueError(f"Invalid probability vectors: {entry['id']}")
            atomic(predictions_dir / f"{entry['id']}.npz", lambda path: np.savez_compressed(
                path, image_id=image_ids, true_label=y, predicted_label=probabilities.argmax(axis=1),
                probabilities=probabilities,
            ))
            print(f"Saved {len(y)} predictions for {entry['id']}", flush=True)

    rows = []
    for entry in manifest["models"]:
        with np.load(predictions_dir / f"{entry['id']}.npz") as prediction:
            y = prediction["true_label"]
            assigned = prediction["probabilities"][np.arange(len(y)), y]
            rows.append({
                "model": entry["id"], "ratio": entry["ratio"], "seed": entry["seed"],
                "accuracy": float(np.mean(prediction["predicted_label"] == y)),
                "log_loss": float(-np.log(np.clip(assigned, np.finfo(assigned.dtype).eps, 1)).mean()),
                "n_images": len(y),
            })
    save_csv(out / f"{manifest['source']}_metrics.csv", rows)


if __name__ == "__main__":
    main()
