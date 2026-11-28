"""Compute paired intervals from saved predictions using only NumPy."""
import numpy as np

from workflow import arguments, output, read_json, save_csv


def paired_intervals(differences, replicates, seed):
    """Reuse image draws across seeds and resample matched training seeds."""
    n_seeds, n_images = differences.shape
    rng = np.random.default_rng(seed)
    per_seed = np.empty((replicates, n_seeds))
    combined = np.empty(replicates)
    for i in range(replicates):
        image_draw = rng.integers(n_images, size=n_images)
        seed_draw = rng.integers(n_seeds, size=n_seeds)
        scores = differences[:, image_draw].mean(axis=1)
        per_seed[i] = scores
        combined[i] = scores[seed_draw].mean()
    return np.percentile(per_seed, [2.5, 97.5], axis=0), np.percentile(combined, [2.5, 97.5])


def main():
    parser = arguments(__doc__)
    parser.add_argument("--directory", type=str, help="Copied output directory containing the manifest and predictions.")
    args = parser.parse_args()
    if args.directory:
        from pathlib import Path

        out = Path(args.directory)
    else:
        out = output(args.smoke)
    manifest = read_json(out / "frozen_models.json")
    source = manifest["source"]
    predictions = out / f"{source}_predictions"
    settings = manifest["bootstrap"]
    correct = {}
    image_ids = labels = None
    for entry in manifest["models"]:
        if entry["ratio"] is None:
            continue
        with np.load(predictions / f"{entry['id']}.npz") as values:
            if image_ids is None:
                image_ids, labels = values["image_id"], values["true_label"]
            if not np.array_equal(image_ids, values["image_id"]) or not np.array_equal(labels, values["true_label"]):
                raise ValueError(f"Misaligned predictions for {entry['id']}")
            correct[entry["ratio"], entry["seed"]] = (values["predicted_label"] == labels).astype(np.int8)
    reference = settings["reference"]
    ratios = sorted({ratio for ratio, _ in correct} - {reference})
    seeds = sorted(seed for ratio, seed in correct if ratio == reference)
    rows = []
    for ratio in ratios:
        differences = np.stack([correct[ratio, seed] - correct[reference, seed] for seed in seeds])
        per_seed, combined = paired_intervals(differences, settings["replicates"], settings["seed"])
        intervals = [
            (seed, differences[i].mean(), *per_seed[:, i]) for i, seed in enumerate(seeds)
        ]
        intervals.append(("pooled", differences.mean(), *combined))
        for seed, estimate, low, high in intervals:
            rows.append({
                "ratio": ratio, "reference": reference, "seed": seed, "difference_pp": estimate * 100,
                "ci95_low_pp": low * 100, "ci95_high_pp": high * 100,
            })
    save_csv(out / f"{source}_paired_bootstrap.csv", rows)


if __name__ == "__main__":
    main()
