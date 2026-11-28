"""Summarize validation fits and write the validation table and key figure."""
import csv
from workflow import arguments, grid, output, read_json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

args = arguments(__doc__).parse_args()
OUT = output(args.smoke)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

with open(OUT / "xgb_runs.csv") as f:
    runs = [{k: float(v) if v else float("nan") for k, v in r.items()} for r in csv.DictReader(f)]
if sorted((r["ratio"], int(r["seed"])) for r in runs) != sorted(grid(args.smoke)):
    raise SystemExit("Finish all validation fits before generating the summary.")
logreg = read_json(OUT / "logreg.json")
max_rounds = read_json(OUT / "xgb_config.json")["max_rounds"]
ratios = sorted({r["ratio"] for r in runs})


def get(ratio, field):
    """Values of one column for one ratio, ordered by seed."""
    rows = sorted((r for r in runs if r["ratio"] == ratio), key=lambda r: r["seed"])
    return np.array([r[field] for r in rows]), [int(r["seed"]) for r in rows]


def mean_sd(x):
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else float("nan"))


# ---------- statistics ----------
stats = {}
acc1, seeds1 = get(1.0, "val_accuracy")
for ratio in ratios:
    acc, seeds = get(ratio, "val_accuracy")
    ll, _ = get(ratio, "val_logloss")
    sec, _ = get(ratio, "fit_seconds")
    rounds, _ = get(ratio, "rounds_run")
    s = {"n": len(acc), "acc": mean_sd(acc), "ll": mean_sd(ll), "sec": mean_sd(sec),
         "rounds": mean_sd(rounds), "ms_per_round": mean_sd(sec / rounds * 1000),
         "cap": int(get(ratio, "hit_cap")[0].sum())}
    common = [x for x in seeds if x in seeds1]  # pair by seed
    if ratio != 1.0 and len(common) > 1:
        d = np.array([acc[seeds.index(x)] - acc1[seeds1.index(x)] for x in common]) * 100  # points
        half = T975[len(d) - 1] * d.std(ddof=1) / np.sqrt(len(d))
        s["diff"] = (d.mean(), d.mean() - half, d.mean() + half)
    if ratio != 1.0 and len(common) > 1:  # same pairing, for log loss (x 1000 to read easily)
        d = np.array([ll[seeds.index(x)] - get(1.0, "val_logloss")[0][seeds1.index(x)] for x in common]) * 1000
        half = T975[len(d) - 1] * d.std(ddof=1) / np.sqrt(len(d))
        s["dll"] = (d.mean(), d.mean() - half, d.mean() + half)
    stats[ratio] = s

# GPU clock vs time per round over all runs (clock is a snapshot taken right after each fit)
ms_round = np.array([r["fit_seconds"] / r["rounds_run"] * 1000 for r in runs])
clock = np.array([r["gpu_clock_mhz"] for r in runs])


# ---------- text summary ----------
lines = [f"{'ratio':>5} {'n':>2} {'val acc':>16} {'val logloss':>16} {'fit time s':>14} {'rounds':>13} {'ms/round':>12} "
         f"{'paired diff vs 1.0 (pts) [95% CI]':>36}"]
for ratio, s in stats.items():
    diff = "-" if "diff" not in s else f"{s['diff'][0]:+.2f} [{s['diff'][1]:+.2f}, {s['diff'][2]:+.2f}]"
    lines.append(f"{ratio:>5} {s['n']:>2} {s['acc'][0]:.4f} +/- {s['acc'][1]:.4f} {s['ll'][0]:.4f} +/- {s['ll'][1]:.4f} "
                 f"{s['sec'][0]:6.1f} +/- {s['sec'][1]:4.1f} {s['rounds'][0]:6.0f} +/- {s['rounds'][1]:4.0f} "
                 f"{s['ms_per_round'][0]:5.0f} +/- {s['ms_per_round'][1]:3.0f} {diff:>36}")
lines.append("paired log-loss difference vs 1.0, x1000 [95% CI]: " + "; ".join(
    f"{r}: {s['dll'][0]:+.2f} [{s['dll'][1]:+.2f}, {s['dll'][2]:+.2f}]" for r, s in stats.items() if "dll" in s))
if np.isfinite(clock).all():
    corr = np.corrcoef(clock, ms_round)[0, 1] if clock.std() and ms_round.std() else float("nan")
    lines.append(f"GPU clock (MHz) after each run: min {clock.min():.0f}, max {clock.max():.0f}; "
                 f"correlation of clock with ms/round over the {len(runs)} runs: {corr:+.2f}")
lines.append(f"logistic regression (C={logreg['best_C']}): val acc {logreg['val_accuracy']:.4f} +/- {logreg['val_se']:.4f} (SE), "
             f"val logloss {logreg['val_logloss']:.4f}")
lines.append(f"runs that hit the {max_rounds}-round cap: {sum(s['cap'] for s in stats.values())}")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------- LaTeX table ----------
tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
       r"Model & Val.\ accuracy & Val.\ log loss & Rounds & Fit time (s) & ms/round & $\Delta$ acc.\ vs.\ 1.0 (pts) \\",
       r"\midrule",
       f"Logistic regression & ${logreg['val_accuracy']:.4f} \\pm {logreg['val_se']:.4f}$\\textsuperscript{{a}} & "
       f"{logreg['val_logloss']:.4f} & -- & -- & -- & -- \\\\", r"\midrule"]
for ratio, s in stats.items():
    label = f"XGBoost, {ratio:.1f}" + (" (baseline)" if ratio == 1.0 else "")
    diff = "--" if "diff" not in s else f"${s['diff'][0]:+.2f}\\ [{s['diff'][1]:+.2f}, {s['diff'][2]:+.2f}]$"
    tex.append(f"{label} & ${s['acc'][0]:.4f} \\pm {s['acc'][1]:.4f}$ & ${s['ll'][0]:.4f} \\pm {s['ll'][1]:.4f}$ & "
               f"{s['rounds'][0]:.0f} & ${s['sec'][0]:.0f} \\pm {s['sec'][1]:.0f}$ & {s['ms_per_round'][0]:.0f} & {diff} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_main.tex").write_text("\n".join(tex) + "\n")

# ---------- key figure ----------
INK, INK2, GRID, BLUE, GRAY = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
x = np.array(ratios)

for ax, field, ylabel in [(axes[0], "val_accuracy", "Validation accuracy"),
                          (axes[1], "fit_seconds", "Training time (s)")]:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for r in ratios:  # individual seeds, small and light
        v, _ = get(r, field)
        ax.plot(np.full(len(v), r) + np.linspace(-0.012, 0.012, len(v)), v, "o", ms=2.5, color=BLUE, alpha=0.35, mew=0)
    m = np.array([get(r, field)[0].mean() for r in ratios])
    sd = np.array([get(r, field)[0].std(ddof=1) if len(get(r, field)[0]) > 1 else 0 for r in ratios])
    ax.errorbar(x, m, yerr=sd, color=BLUE, lw=1.4, marker="o", ms=4.5, capsize=2.5, elinewidth=1)
    ax.plot([1.0], [m[-1]], "o", ms=8, mfc="none", mec=INK, mew=1)  # highlight the full-data baseline
    ax.set_xticks(x)
    ax.set_xlim(0.15, 1.06)
    ax.set_xlabel("Subsample ratio")
    ax.set_ylabel(ylabel)

a = axes[0]
a.axhline(logreg["val_accuracy"], color=GRAY, lw=1.1, ls="--")
a.text(0.16, logreg["val_accuracy"] + 0.0012, f"logistic regression ({logreg['val_accuracy']:.3f})", color=INK2, fontsize=7)
a.annotate("full-data baseline\n(ratio 1.0)", xy=(1.0, stats[1.0]["acc"][0] - 0.0012), xytext=(0.98, 0.85), textcoords="axes fraction",
           color=INK2, fontsize=7, ha="right", va="top", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
axes[0].set_title("(a) accuracy", loc="left", fontsize=8)
axes[1].set_title("(b) training time", loc="left", fontsize=8)
axes[1].set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT / "key_figure.pdf")
fig.savefig(OUT / "key_figure.png", dpi=200)
print("wrote", OUT / "key_figure.pdf")
