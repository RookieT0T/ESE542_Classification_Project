"""Step 1.4: time ONE XGBoost fit (subsample = 1.0) on the full 50,000 training images.

Purpose: find out whether 5 ratios x 5 seeds is affordable (25 fits ~ 25 x this time).
Everything runs on the GPU: the data lives in GPU memory (CuPy arrays), XGBoost
trains with device="cuda", and the predictions are computed and compared on the GPU.
"""
import time

import cupy as cp
from xgboost import XGBClassifier

from common import SEED, load_train_val

X_train, y_train, X_val, y_val = (cp.asarray(a) for a in load_train_val())  # to GPU memory


def make_model(n_rounds):
    return XGBClassifier(
        device="cuda",
        tree_method="hist",          # fast histogram-based split search
        max_bin=64,                  # 64 histogram bins per pixel instead of 256: ~30% faster, same accuracy
        n_estimators=n_rounds,       # upper limit; early stopping picks the real number
        learning_rate=0.1,
        subsample=1.0,               # every row is used for every tree
        early_stopping_rounds=20,    # stop when validation loss has not improved for 20 rounds
        eval_metric="mlogloss",
        random_state=SEED,
    )


# Warm-up: the first GPU call pays a one-off start-up cost; don't count it.
make_model(1).fit(X_train[:500], y_train[:500], eval_set=[(X_val[:500], y_val[:500])], verbose=False)

model = make_model(1000)
start = time.perf_counter()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
cp.cuda.Stream.null.synchronize()  # make sure the GPU work is finished before stopping the clock
seconds = time.perf_counter() - start

acc = float((cp.asarray(model.predict(X_val)) == y_val).mean())
se = (acc * (1 - acc) / len(y_val)) ** 0.5

print(f"fit time ({len(y_train)} rows) : {seconds:.1f} s")
print(f"best round              : {model.best_iteration}")
print(f"validation accuracy     : {acc:.4f} +/- {se:.4f}")
print(f"\nEstimated 25 fits (5 ratios x 5 seeds): {25 * seconds / 60:.0f} min")
