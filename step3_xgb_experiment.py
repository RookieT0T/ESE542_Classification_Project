"""Main experiment: XGBoost with 5 subsample ratios x 5 seeds = 25 fits, all on the GPU.

Every fit uses the same fixed settings except `subsample` and the seed. The number of boosting
rounds is chosen per fit by early stopping on the 10,000 validation images.
The runs are shuffled, so GPU heating (and the clock slowdown it causes) is spread evenly
across ratios. Each finished run is appended to results/xgb_runs.csv, so an interrupted job
can be restarted and continues where it stopped. The test set is not used.
"""
import csv
import random
import subprocess
import time
from pathlib import Path

import cupy as cp
from xgboost import XGBClassifier

from common import SEED, load_train_val

RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MAX_ROUNDS = 1000
OUT = Path(__file__).resolve().parent / "results"
(OUT / "models").mkdir(parents=True, exist_ok=True)
CSV = OUT / "xgb_runs.csv"
FIELDS = ["ratio", "seed", "fit_seconds", "rounds_run", "best_round", "hit_cap",
          "val_accuracy", "val_logloss", "gpu_temp_c", "gpu_clock_mhz"]

X_train, y_train, X_val, y_val = (cp.asarray(a) for a in load_train_val())  # to GPU memory


def make_model(ratio, seed, n_rounds=MAX_ROUNDS):
    return XGBClassifier(
        device="cuda",
        tree_method="hist",
        max_bin=64,                  # fixed for all runs
        n_estimators=n_rounds,       # upper limit; early stopping picks the real number
        learning_rate=0.1,
        subsample=ratio,             # <- the quantity under study
        early_stopping_rounds=20,    # stop when validation log loss has not improved for 20 rounds
        eval_metric="mlogloss",
        random_state=seed,           # seeds the random choice of rows for each tree
    )


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [round(float(v)) for v in out.split(",")]  # nvidia-smi may print e.g. "86.10"


# Which runs are still to do (resume support)
done = set()
if CSV.exists():
    with open(CSV) as f:
        done = {(float(r["ratio"]), int(r["seed"])) for r in csv.DictReader(f)}
runs = [(r, s) for r in RATIOS for s in SEEDS if (r, s) not in done]
random.Random(SEED).shuffle(runs)
print(f"{len(done)} runs already done, {len(runs)} to do", flush=True)

# Warm-up: the first GPU call pays a one-off start-up cost; don't count it.
make_model(1.0, 0, 1).fit(X_train[:500], y_train[:500], eval_set=[(X_val[:500], y_val[:500])], verbose=False)

if not CSV.exists():
    with open(CSV, "w", newline="") as f:
        csv.DictWriter(f, FIELDS).writeheader()

for i, (ratio, seed) in enumerate(runs, 1):
    model = make_model(ratio, seed)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - start

    history = model.evals_result()["validation_0"]["mlogloss"]
    temp, clock = gpu_state()
    row = {
        "ratio": ratio, "seed": seed, "fit_seconds": round(seconds, 2),
        "rounds_run": len(history), "best_round": model.best_iteration,
        "hit_cap": int(len(history) >= MAX_ROUNDS),
        "val_accuracy": float((cp.asarray(model.predict(X_val)) == y_val).mean()),
        "val_logloss": history[model.best_iteration],
        "gpu_temp_c": temp, "gpu_clock_mhz": clock,
    }
    with open(CSV, "a", newline="") as f:
        csv.DictWriter(f, FIELDS).writerow(row)
    model.save_model(OUT / "models" / f"ratio{ratio}_seed{seed}.ubj")  # for the one-time test evaluation
    print(f"[{i}/{len(runs)}] ratio={ratio} seed={seed}  {seconds:5.1f}s  rounds={len(history)}  "
          f"val_acc={row['val_accuracy']:.4f}  gpu {temp}C {clock}MHz", flush=True)
