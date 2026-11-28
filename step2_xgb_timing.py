"""Benchmark fixed rounds with/without validation on the same host."""
import random
import time
from statistics import mean, stdev

from workflow import RATIOS, arguments, data, host, output, read_json, record_config, save_csv, save_json
from xgb_setup import device_arrays, fitted_config, gpu_state, model, xgb


def main():
    args = arguments(__doc__).parse_args()
    out = output(args.smoke)
    devices = ("cpu",) if args.smoke else ("cpu", "cuda")
    ratios = (0.4, 1.0) if args.smoke else RATIOS
    rounds, repeats, threads = (4, 2, 2) if args.smoke else (100, 5, 4)
    config = {
        **host(), "xgboost": xgb.__version__, "rounds": rounds, "repeats": repeats,
        "devices": list(devices), "ratios": list(ratios), "threads": threads,
    }
    if "cuda" in devices:
        state = gpu_state()
        config.update(gpu=state["gpu"], driver=state["driver"])
    record_config(out / "timing_config.json", config)
    arrays = data(args.smoke)
    raw = {
        path.stem: read_json(path)
        for path in (out / "timing").glob("*.json")
    }
    paired = []
    for device in devices:
        complete = all(
            f"{device}_{ratio}_{repeat}_{int(score)}" in raw
            for ratio in ratios for repeat in range(repeats) for score in (False, True)
        )
        if complete:
            print(f"{device} timing runs already saved", flush=True)
        else:
            (X, y, X_val, y_val), sync = device_arrays(device, arrays)
            model(device, 1.0, 0, 1, threads=threads).fit(
                X[:500], y[:500], eval_set=[(X_val, y_val)], verbose=False,
            )
        jobs = [(ratio, repeat) for ratio in ratios for repeat in range(repeats)]
        random.Random(0).shuffle(jobs)
        for ratio, repeat in jobs:
            order = [False, True] if repeat % 2 == 0 else [True, False]
            times = {}
            for score in order:
                path = out / "timing" / f"{device}_{ratio}_{repeat}_{int(score)}.json"
                if path.stem in raw:
                    row = raw[path.stem]
                else:
                    estimator = model(device, ratio, repeat, rounds, threads=threads)
                    sync()
                    start = time.perf_counter()
                    estimator.fit(X, y, eval_set=[(X_val, y_val)] if score else None, verbose=False)
                    sync()
                    elapsed = time.perf_counter() - start
                    fitted_config(estimator, device)
                    row = {
                        "device": device, "ratio": ratio, "repeat": repeat, "seed": repeat,
                        "validation": score, "rounds": estimator.get_booster().num_boosted_rounds(),
                        "fit_seconds": elapsed, "seconds_per_round": elapsed / rounds,
                    }
                    save_json(path, row)
                    print(row, flush=True)
                raw[path.stem] = row
                times[score] = row["seconds_per_round"] * 1000
                save_csv(out / "timing_runs.csv", list(raw.values()))
            paired.append({
                "device": device, "ratio": ratio, "repeat": repeat,
                "training_ms_per_round": times[False], "with_validation_ms_per_round": times[True],
                "validation_overhead_ms_per_round": times[True] - times[False],
            })
            save_csv(out / "timing_pairs.csv", paired)
    summary = []
    for device in devices:
        for ratio in ratios:
            group = [row for row in paired if row["device"] == device and row["ratio"] == ratio]
            metrics = {}
            for key in ("training_ms_per_round", "with_validation_ms_per_round", "validation_overhead_ms_per_round"):
                values = [row[key] for row in group]
                metrics[f"mean_{key}"] = mean(values)
                metrics[f"sd_{key}"] = stdev(values)
            summary.append({"device": device, "ratio": ratio, "repeats": repeats, **metrics})
    save_csv(out / "timing_summary.csv", summary)


if __name__ == "__main__":
    main()
