# XGBoost subsampling on Fashion-MNIST

The original stratified split (seed 0) uses 50,000 official training images for
fitting and 10,000 for validation. The official 10,000 test images are loaded
only by the explicit final scoring command. The draft outputs are in
`results/`; new runs go in `results/final/` or `results/smoke/`.

## Setup

Clone [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) as
`fashion-mnist/` in this project, or copy the dataset and loader to the same
paths. Use the same CUDA-enabled PyTorch, XGBoost, and CuPy environment as the
draft for the final runs. Use the CuPy build that matches the NVIDIA machine's
CUDA environment. Python packages used are NumPy, scikit-learn, XGBoost,
PyTorch, and Matplotlib, plus CuPy on the NVIDIA machine. On Apple Silicon,
use an ARM64 Python and install OpenMP (for example, `brew install libomp`).

## MacBook smoke test

```bash
python smoke_test.py
```

The smoke test uses the same scripts on CPU with 2,000 rows drawn from the
original fitting partition, 500 rows from the original validation partition,
and the full five-fold logistic regression grid. It runs XGBoost at 0.4 and
1.0 with two seeds and up to eight rounds, plus two paired timing repeats
per ratio with four fixed rounds. Prediction export and bootstrap use the
500 validation images. These measurements only verify the workflow.

## Final runs on the NVIDIA computer

Run on an idle machine, in sequence, from the project folder. Keep the same
environment and machine when resuming. Run the CPU and GPU timings on this
same machine.

```bash
python step1_logreg.py
python step2_xgb_timing.py
python step3_xgb_experiment.py
python step4_analyze.py
python step5_score.py --freeze
```

The final grid is five ratios by ten seeds. The final output starts a fresh
50-fit run, since the five-seed draft CSV lacks the per-run learning curves
and complete model configuration. Each fit saves its checkpoint and JSON
record before updating the CSV. Restart the same command to resume.
The timing benchmark runs 100 fixed rounds, with early stopping disabled,
for five paired repeats per ratio per device. The fit with validation turned
off measures training and fit setup, including data matrix construction.
The paired time difference estimates the extra work of including validation,
including its setup cost. Small differences can be noisy.

Once the validation work is complete and the frozen manifest has been
reviewed, run this held-out evaluation once:

```bash
python step5_score.py --official-test
```

Scoring verifies the frozen records and checkpoint hashes and uses each fit's
frozen best-round limit. A retry picks up missing prediction files. Completed
prediction files are reused.

## Results and Mac bootstrap

| Output | Contents |
|---|---|
| `logreg.json`, `logreg_folds/`, `logreg.pt` | Five fold accuracies per C, CV mean and SD, chosen C, solver iterations, validation metrics, fitted weights |
| `xgb_runs.csv`, `xgb/`, `models/` | Per-run validation metrics and timing, zero-based best iteration, one-based prediction round count, rounds actually run, validation loss curves, effective booster configuration, and checkpoints |
| `timing_runs.csv`, `timing_pairs.csv`, `timing_summary.csv` | Individual fixed-round measurements, paired validation overhead, and mean/SD by device and ratio |
| `*_config.json`, `frozen_models.json` | Run environment and settings, model hashes, scoring round limits, and bootstrap settings |
| `summary.txt`, `table_main.tex`, `key_figure.pdf` | Validation table, seed-only intervals, and figure |
| `official_test_predictions/`, `official_test_metrics.csv` | One compressed NumPy file per model with aligned image IDs, labels, predictions, and ten probabilities per image, plus per-model metrics |

Copy `frozen_models.json` and `official_test_predictions/` into one
directory on the Mac. NumPy is the only third-party dependency for inference
from saved predictions:

```bash
python step6_bootstrap.py --directory /path/to/copied-results
```

This creates `official_test_paired_bootstrap.csv` with image-paired intervals
for each seed and a pooled interval that resamples matched seeds and shared
images, all relative to ratio 1.0. Accuracy differences are in percentage
points. Smoke scoring and bootstrap create the corresponding
`validation_*` outputs.
