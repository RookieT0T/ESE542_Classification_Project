"""Run the CPU smoke profile using the same training, scoring, and analysis scripts."""
import subprocess
import sys

from workflow import ROOT


def main():
    for script in (
        "step1_logreg.py", "step2_xgb_timing.py", "step3_xgb_experiment.py",
        "step4_analyze.py", "step5_score.py", "step6_bootstrap.py",
    ):
        subprocess.run([sys.executable, str(ROOT / script), "--smoke"], check=True)


if __name__ == "__main__":
    main()
