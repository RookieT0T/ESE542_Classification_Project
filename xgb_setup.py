"""Common model settings for early-stopped fits and the timing benchmark."""
import json
import subprocess

import xgboost as xgb


def model(device, ratio, seed, rounds, *, early_stopping=None, threads=None):
    return xgb.XGBClassifier(
        device=device, tree_method="hist", max_bin=64, max_depth=6,
        learning_rate=0.1, subsample=ratio, sampling_method="uniform",
        n_estimators=rounds, early_stopping_rounds=early_stopping,
        eval_metric="mlogloss", random_state=seed, n_jobs=threads,
    )


def gpu_state():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm",
         "--format=csv,noheader,nounits"], text=True,
    ).splitlines()[0]
    name, driver, temperature, clock = (item.strip() for item in result.split(","))
    return {"gpu": name, "driver": driver, "gpu_temp_c": float(temperature), "gpu_clock_mhz": float(clock)}


def device_arrays(device, arrays):
    if device == "cuda":
        import cupy as cp

        return tuple(cp.asarray(array) for array in arrays), cp.cuda.Device().synchronize
    return arrays, lambda: None


def fitted_config(estimator, device):
    config = json.loads(estimator.get_booster().save_config())
    actual = config["learner"]["generic_param"]["device"]
    if actual.split(":")[0] != device:
        raise RuntimeError(f"Requested {device}, but XGBoost fit ran on {actual}")
    return config
