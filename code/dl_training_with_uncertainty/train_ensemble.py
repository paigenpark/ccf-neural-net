"""Train one DL cohort-fertility configuration as a multi-seed ensemble.

HOW TO USE THIS FILE
--------------------
1. Edit the CONFIG block below -- the seven hyperparameters and VARIANT name.
2. Give VARIANT a new name for every distinct config, so runs never overwrite
   each other and the eval notebook can plot them side by side.
3. Run it from the repo root, naming the log after VARIANT so concurrent or
   successive runs don't clobber each other's output. TensorFlow lives in the
   project venv, so invoke that interpreter explicitly -- a bare `python` will
   fail on `import tensorflow`:

       nohup .venv/bin/python code/dl_training_with_uncertainty/train_ensemble.py \
           > train_CCF_Net_u110_l4_d0.0_w40.log 2>&1 &

Each run trains one model per jump-off year (29 of them, 1985-2013) per seed,
in two phases of a fixed pipeline.EPOCHS epochs each (mean network, then a
variance head on the frozen network) -- so wall time is predictable and scales
with 29 x len(SEEDS). Trim SEEDS while iterating.

UNCERTAINTY
-----------
Each member predicts a mean and a variance per cell. The ensemble is treated as
an equal-weight Gaussian mixture, so per cell:

    Rate         = mean of the member means
    VarAleatoric = mean of the member variances
    VarEpistemic = variance of the member means across seeds
    Var          = VarAleatoric + VarEpistemic,  SD = sqrt(Var)
    Lower95/Upper95 = Rate -/+ 1.96 * SD, clipped to [0, 1]

These are bounds on individual age-specific rates. They say nothing about
completed cohort fertility, which needs the correlation across ages.

WHY ENSEMBLE
------------
A single training run of this model has large seed variance -- per-jump-off-year
RMSE swings by roughly 2x across seeds, and mean per-cell forecast SD is ~0.007.
Averaging several seeds gives both a more accurate forecast and a stable series
to report. The per-cell SD across seeds is written out too, as a spread band.

OUTPUTS (all under ../../data/<OUT_SUBDIR>)
-------------------------------------------
    dl_forecasts_<VARIANT>.csv          ensemble mean forecast (cohort rates)
    dl_obs_cohort_<VARIANT>.csv         observed cohort rates (seed-independent)
    dl_forecasts_<VARIANT>_uncertainty.csv  per-cell variance and 95% bounds
    dl_forecasts_<VARIANT>_seedsd.csv   per-cell SD across seeds
    dl_forecasts_<VARIANT>_seed<K>.csv  each individual member (Rate and Var)
    dl_config_<VARIANT>.json            the exact config that produced the above

To plot a run, add it to the registry in code/evaluation/eval_figures_cohort.ipynb.
The notebook resolves names as '<DATA_DIR>/<name>.csv', so carry OUT_SUBDIR in the
name -- no notebook plumbing needed:

    '<label>': dict(pred='<OUT_SUBDIR>/dl_forecasts_<VARIANT>',
                    obs='<OUT_SUBDIR>/dl_obs_cohort_<VARIANT>',
                    color='#1f77b4', marker='s', in_common=False),
"""

import json
import os
import signal
import sys

# Keep the process alive when backgrounded. Under shell job control a background
# process that writes to the terminal gets stopped with SIGTTOU and shows up as
# "suspended (tty output)" -- TensorFlow can do this during import or fit even
# with stdout redirected. Ignoring these signals lets it proceed; all real output
# already goes to the log file. Must be set before TensorFlow is imported.
signal.signal(signal.SIGTTOU, signal.SIG_IGN)
signal.signal(signal.SIGTTIN, signal.SIG_IGN)

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline  # noqa: E402

# ===========================================================================
# CONFIG -- this is the part you edit
# ===========================================================================

# Name this run.
VARIANT = "CCF_Net_u110_l4_d0.0_w40"  # <--- EDIT THIS TO SAVE OTHER VARIANTS UNDER A DIFFERENT NAME

# Hyperparameters. Set before any evaluation and deliberately not tuned.
# Can be adjusted to test a handful of variants of CCF-Net
UNITS = 110            # width of each hidden layer
N_LAYERS = 4           # number of hidden layers (before the output layer)
DROPOUT = 0.0          # dropout rate after each hidden layer, 0.0 = off
TRAIN_WINDOW = 40      # years of history to train on, counting back from each
                       # jump-off year. None = use all available history.
                       # e.g. 40 -> for JOY 2000, train on 1961-2000.
AGE_EMBED_DIM = 5      # size of the learned age embedding
GEO_EMBED_DIM = 5      # size of the learned country embedding
LEARNING_RATE = 1e-3   # Adam learning rate

# Seeds to ensemble over. Identical data and hyperparameters -- only weight
# init and batch sampling differ. More seeds = less noise, more time.
SEEDS = [0, 1, 2, 3, 4]

# Which Lexis shape to read: "square" (completed age, ACY) or
# "parallelogram" (ARDY). Not a tuning knob -- it changes the input data.
LEXIS_SHAPE = "square"

# Subfolder of ../../data to write this run's outputs into, so exploratory runs
# don't crowd the top-level data dir. "" writes directly to ../../data.
OUT_SUBDIR = ""


# ===========================================================================
# Everything below here runs the variant; you shouldn't need to edit it.
# ===========================================================================

HPARAMS = dict(
    units=UNITS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    age_embed_dim=AGE_EMBED_DIM,
    geo_embed_dim=GEO_EMBED_DIM,
    learning_rate=LEARNING_RATE,
)

DATA_DIR = pipeline.DATA_DIR
OUT_DIR = os.path.join(DATA_DIR, OUT_SUBDIR) if OUT_SUBDIR else DATA_DIR
CELL_KEY = ["Country", "Year", "Age", "JumpOffYear"]
Z_95 = 1.96            # normal quantile for the 95% bounds
# Per-member models are disposable (the seed plus this config regenerates them),
# so they go under the gitignored data/ dir.
MEMBER_MODELS_ROOT = os.path.join(OUT_DIR, "ensemble_member_models", VARIANT)


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def save_config():
    """Record the exact config next to the outputs, so a CSV is never orphaned."""
    path = out_path(f"dl_config_{VARIANT}.json")
    with open(path, "w") as f:
        json.dump({
            "variant": VARIANT,
            "hparams": HPARAMS,
            "train_window": TRAIN_WINDOW,
            "lexis_shape": LEXIS_SHAPE,
            "seeds": SEEDS,
            "jump_off_years": pipeline.JUMP_OFF_YEARS,
            "age_range": [pipeline.AGE_MIN, pipeline.AGE_MAX],
            "forecast_len": pipeline.FORECAST_LEN,
            "epochs": pipeline.EPOCHS,
            "passes_per_epoch": pipeline.PASSES_PER_EPOCH,
            "total_passes": pipeline.TOTAL_PASSES,
            "batch_size": pipeline.BATCH_SIZE,
            "training": "freeze ensemble: mean phase (MSE), then variance "
                        "head on the frozen network (Gaussian NLL); "
                        "'epochs' is the budget for each phase",
        }, f, indent=2)
    print(f"Config recorded: {path}")


def train_member(seed, asfr, geo_dim, valid_countries, valid_combos):
    """Train every jump-off year under one seed; return its cohort forecast."""
    predicted_rates = pipeline.train_and_forecast(
        asfr, geo_dim, valid_countries, valid_combos,
        hparams=HPARAMS,
        train_window=TRAIN_WINDOW,
        seed=seed,
        models_dir=os.path.join(MEMBER_MODELS_ROOT, f"seed{seed}"),
        tag=f"{VARIANT}_seed{seed}",
    )
    member = pipeline.build_predicted_cohort(predicted_rates)
    member.to_csv(out_path(f"dl_forecasts_{VARIANT}_seed{seed}.csv"), index=False)
    print(f"[seed {seed}] member forecast: {len(member):,} rows")
    return member


def report_coverage(unc, obs):
    """Print the share of observed post-JOY rates inside the 95% bounds.

    Reporting only, like the validation loss: a well-calibrated interval covers
    about 95%. Only cells whose forecast year has since been observed count.
    """
    obs_key = ["Country", "Year", "Age"]   # obs has one row per cell, no JOY
    both = unc.merge(obs[obs_key + ["Rate"]], on=obs_key, suffixes=("", "Obs"))
    if both.empty:
        print("95% coverage: no observed rates over the forecast horizon")
        return
    both["Inside"] = both["RateObs"].between(both["Lower95"], both["Upper95"])
    print("Empirical coverage of the 95% bounds (observed forecast cells):")
    for joy, g in both.groupby("JumpOffYear"):
        print(f"  JOY {joy}: {g['Inside'].mean():.1%} of {len(g):,} cells")
    print(f"  overall : {both['Inside'].mean():.1%} of {len(both):,} cells")


def main():
    print(f"Variant '{VARIANT}'")
    print(f"  hparams      : {HPARAMS}")
    print(f"  train_window : {TRAIN_WINDOW}")
    print(f"  lexis_shape  : {LEXIS_SHAPE}")
    print(f"  seeds        : {SEEDS}\n")
    save_config()

    asfr = pipeline.load_data(LEXIS_SHAPE)
    geo_dim, valid_countries, valid_combos = pipeline.validate_countries(asfr)

    # Observed rates don't depend on the model -- build once, save under this
    # variant's name so the notebook can pair pred + obs consistently.
    obs = pipeline.build_observed_cohort(asfr, valid_combos)
    obs.to_csv(out_path(f"dl_obs_cohort_{VARIANT}.csv"), index=False)

    members = []
    for seed in SEEDS:
        print(f"\n{'#' * 60}\n# MEMBER  seed={seed}\n{'#' * 60}")
        members.append(train_member(seed, asfr, geo_dim, valid_countries,
                                    valid_combos).assign(_seed=seed))

    everything = pd.concat(members, ignore_index=True)
    grouped = everything.groupby(CELL_KEY)   # all members share one grid

    # Predictive uncertainty: the ensemble as an equal-weight Gaussian mixture.
    # Total variance = mean member variance (aleatoric) + variance of the member
    # means (epistemic; ddof=0, the mixture's own variance).
    unc = pd.DataFrame({
        "Rate": grouped["Rate"].mean(),
        "VarAleatoric": grouped["Var"].mean(),
        "VarEpistemic": grouped["Rate"].var(ddof=0),
    }).reset_index()

    # Ensemble point forecast = the mixture mean, in the shared CSV schema.
    ens = unc[CELL_KEY + ["Rate"]].copy()
    ens["Method"] = pipeline.METHOD_NAME
    ens["Key"] = ens["Method"] + "_" + ens["Country"].astype(str)
    ens = ens[["Country", "Year", "Age", "Rate", "JumpOffYear", "Method", "Key"]]
    ens.to_csv(out_path(f"dl_forecasts_{VARIANT}.csv"), index=False)

    # Per-cell disagreement across seeds (sample SD, ddof=1) -> spread band for plots.
    sd = grouped["Rate"].std().rename("RateSD").reset_index()
    sd.to_csv(out_path(f"dl_forecasts_{VARIANT}_seedsd.csv"), index=False)

    unc["Var"] = unc["VarAleatoric"] + unc["VarEpistemic"]
    unc["SD"] = unc["Var"] ** 0.5
    unc["Lower95"] = (unc["Rate"] - Z_95 * unc["SD"]).clip(lower=0.0)
    unc["Upper95"] = (unc["Rate"] + Z_95 * unc["SD"]).clip(upper=1.0)
    unc.to_csv(out_path(f"dl_forecasts_{VARIANT}_uncertainty.csv"), index=False)

    print(f"\n{'=' * 60}")
    print(f"Ensemble '{VARIANT}' over seeds {SEEDS}: {len(ens):,} cells")
    print(f"Mean per-cell seed SD: {sd['RateSD'].mean():.5f}")
    print(f"Mean per-cell predictive SD: {unc['SD'].mean():.5f} "
          f"(aleatoric share of variance: "
          f"{unc['VarAleatoric'].sum() / unc['Var'].sum():.1%})")
    report_coverage(unc, obs)
    print(f"Saved dl_forecasts_{VARIANT}.csv, dl_obs_cohort_{VARIANT}.csv, "
          f"dl_forecasts_{VARIANT}_uncertainty.csv, "
          f"dl_forecasts_{VARIANT}_seedsd.csv, per-seed members, and "
          f"dl_config_{VARIANT}.json")


if __name__ == "__main__":
    main()
