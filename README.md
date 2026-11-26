# XGBoost subsample ratio on Fashion-MNIST

Effect of XGBoost's `subsample` ratio on Fashion-MNIST accuracy and training time.
Everything runs on an NVIDIA GPU. The current laptop GPU is for experimenting; we will
switch to a faster GPU later if the timing budget gets tight.

## Files

| File | Purpose |
|---|---|
| `common.py` | Loads the 60,000 training images and splits them 50,000 train / 10,000 validation (seed 0). The test set is not loaded. |
| `step1_logreg.py` | Logistic regression baseline (PyTorch): picks `C` by 5-fold CV on the training images. |
| `step2_xgb_timing.py` | Times one XGBoost fit on the 50,000 training images. |
| `step3_xgb_experiment.py` | Main experiment: 5 subsample ratios x 5 seeds = 25 fits with early stopping. |
| `step4_analyze.py` | Summary statistics, LaTeX table and key figure from the results. |

## Setup (once, from the project folder)

```bash
git clone https://github.com/zalandoresearch/fashion-mnist   # dataset + loader (not stored in this repo)
pip install --user --break-system-packages uv
export PATH=$HOME/.local/bin:$PATH
uv venv .venv
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python \
    numpy scikit-learn xgboost matplotlib torch cupy-cuda13x
```

## Run

```bash
.venv/bin/python step1_logreg.py       # ~20 s
.venv/bin/python step2_xgb_timing.py   # ~1 min
.venv/bin/python step3_xgb_experiment.py   # ~25 min; resumes if interrupted
.venv/bin/python step4_analyze.py      # writes the summary, table and figure to results/
```

Run one script at a time; sharing the GPU distorts the timings.
