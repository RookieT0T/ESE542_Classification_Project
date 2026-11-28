"""Fit each subsample/seed combination and checkpoint its model and measurements."""
import random
import time

from workflow import arguments, atomic, data, grid, host, output, read_json, record_config, save_csv, save_json
from xgb_setup import device_arrays, fitted_config, gpu_state, model, xgb


def main():
    args = arguments(__doc__).parse_args()
    out = output(args.smoke)
    device = "cpu" if args.smoke else "cuda"
    rounds, patience, threads = (8, 3, 2) if args.smoke else (1000, 20, None)
    runs = grid(args.smoke)
    random.Random(0).shuffle(runs)
    config = {
        **host(), "xgboost": xgb.__version__, "device": device, "threads": threads,
        "max_rounds": rounds, "patience": patience, "smoke": args.smoke,
    }
    if device == "cuda":
        state = gpu_state()
        config.update(gpu=state["gpu"], driver=state["driver"])
    record_config(out / "xgb_config.json", config)
    rows = {}
    pending = []
    for ratio, seed in runs:
        name = f"ratio{ratio}_seed{seed}"
        checkpoint = out / "models" / f"{name}.ubj"
        record = out / "xgb" / f"{name}.json"
        if record.exists() and checkpoint.exists():
            rows[name] = read_json(record)["metrics"]
        else:
            pending.append((ratio, seed, name, checkpoint, record))
    save_csv(out / "xgb_runs.csv", list(rows.values()))
    if not pending:
        print(f"All {len(runs)} fits already saved in {out}")
        return

    (X, y, X_val, y_val), sync = device_arrays(device, data(args.smoke))
    model(device, 1.0, 0, 1, threads=threads).fit(X[:500], y[:500], verbose=False)
    for ratio, seed, name, checkpoint, record in pending:
        estimator = model(device, ratio, seed, rounds, early_stopping=patience, threads=threads)
        sync()
        start = time.perf_counter()
        estimator.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        sync()
        elapsed = time.perf_counter() - start
        booster_config = fitted_config(estimator, device)
        state = gpu_state() if device == "cuda" else {}
        curves = estimator.evals_result()
        losses = curves["validation_0"]["mlogloss"]
        best = estimator.best_iteration
        predictions = estimator.predict(X_val, iteration_range=(0, best + 1))
        if device == "cuda":
            import cupy as cp

            predictions = cp.asarray(predictions)
        metrics = {
            "ratio": ratio, "seed": seed, "fit_seconds": elapsed,
            "rounds_run": len(losses), "best_round": best, "prediction_rounds": best + 1,
            "hit_cap": int(len(losses) == rounds), "seconds_per_round": elapsed / len(losses),
            "val_accuracy": float((predictions == y_val).mean()), "val_logloss": losses[best],
            "gpu_temp_c": state.get("gpu_temp_c", ""), "gpu_clock_mhz": state.get("gpu_clock_mhz", ""),
        }
        atomic(checkpoint, estimator.save_model)
        save_json(record, {
            "metrics": metrics, "learning_curves": curves, "booster_config": booster_config,
            "model_path": str(checkpoint.relative_to(out)), "n_train": len(y), "n_val": len(y_val),
        })
        rows[name] = metrics
        save_csv(out / "xgb_runs.csv", list(rows.values()))
        print(f"[{len(rows)}/{len(runs)}] {name}: {metrics['val_accuracy']:.4f}, {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
