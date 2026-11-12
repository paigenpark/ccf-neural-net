"""Run the CCF-Net ensemble from a future jump-off year, without editing the trainer.

Called by run_all.py. To run the DL model on its own (from the repo root):

    .venv/bin/python code/future_forecasts/run_dl_future.py --joys 2019

The trainer (code/dl_training_with_uncertainty/) keeps its settings as module
globals that are read when the code runs. This script imports it, changes the
jump-off year and output location, and calls train_ensemble.main(). The
hyperparameters, seeds and ensembling all come from train_ensemble.py unchanged.

pipeline.YEAR_MAX
-----------------
YEAR_MAX = 2015 has two jobs in pipeline.py:
  1. it is the upper anchor of the year scaling in normalize_year(), and
  2. validate_countries() skips any jump-off year after it.
Left alone, (2) would drop every country at a 2019 jump-off. So this script
pins the year scaling to 1950-2015 (the same inputs the main run saw; forecast
years map above 1.0, as they already do in the main run) and then raises
YEAR_MAX to the jump-off year. validate_countries() still drops countries whose
series has gaps before the jump-off.

Outputs (named as train_ensemble.py names them, with VARIANT = future_joy<J>):

    dl_forecasts_future_joy<J>.csv               ensemble forecast (cohort rates)
    dl_obs_cohort_future_joy<J>.csv              observed cohort rates
    dl_forecasts_future_joy<J>_uncertainty.csv   per-cell variance and 95% bounds
    dl_forecasts_future_joy<J>_seedsd.csv        per-cell SD across seeds
    dl_forecasts_future_joy<J>_seed<K>.csv       each ensemble member
    dl_config_future_joy<J>.json                 the config used
"""

import argparse
import os
import sys

DL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../dl_training_with_uncertainty"))
sys.path.insert(0, DL_DIR)

import pipeline          # noqa: E402
import train_ensemble    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joys", default="2019", help="comma-separated jump-off year(s)")
    ap.add_argument("--out-dir", default=None,
                    help="absolute output directory (default: data/future_forecasts)")
    args = ap.parse_args()

    joys = [int(j) for j in args.joys.split(",")]
    out_dir = args.out_dir or os.path.join(pipeline.DATA_DIR, "future_forecasts")

    # Pin the year scaling to its original range BEFORE changing YEAR_MAX.
    anchor_min, anchor_max = pipeline.YEAR_MIN, pipeline.YEAR_MAX
    pipeline.normalize_year = lambda year: (year - anchor_min) / (anchor_max - anchor_min)

    pipeline.JUMP_OFF_YEARS = joys
    pipeline.YEAR_MAX = max(joys)

    variant = "future_joy" + "_".join(str(j) for j in joys)
    train_ensemble.VARIANT = variant
    train_ensemble.OUT_DIR = out_dir
    train_ensemble.MEMBER_MODELS_ROOT = os.path.join(out_dir, "ensemble_member_models", variant)

    print("DL future forecast")
    print(f"  jump-off year(s) : {joys}")
    print(f"  forecasting      : {min(joys) + 1}-{max(joys) + pipeline.FORECAST_LEN}")
    print(f"  year scaling     : {anchor_min}-{anchor_max} (unchanged)")
    print(f"  variant          : {variant}")
    print(f"  output           : {out_dir}\n")

    train_ensemble.main()


if __name__ == "__main__":
    main()
