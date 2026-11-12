"""Cohort ASFR forecasting pipeline

This module holds the machinery: load data, decide which countries are usable,
build the network, train it per jump-off year, and convert period rates to
cohort rates. Nothing here reads environment variables or config files -- every
setting arrives as an explicit argument from the calling script. If you want to
change a hyperparameter, change it in train_ensemble.py, not here.

The pieces, in the order they run:

    load_data()             read the ASFR file, keep ages 15-44
    validate_countries()    drop countries with missing ages / year gaps
    train_and_forecast()    per jump-off year: train for EPOCHS epochs on the
                            train_window years up to the JOY (mean network, then a variance head on
                            the frozen network), then predict the mean and
                            variance of each rate for JOY+1..JOY+30
    build_predicted_cohort()/build_observed_cohort()
                            reindex period rates by birth cohort for the eval notebook

Terminology: a "jump-off year" (JOY) is the last year of observed data the model
is allowed to see. Everything after it is forecast, and is what gets graded.

Training uses only data <= JOY, for a fixed EPOCHS epochs. The validation set
is the observed post-JOY window -- i.e. the graded period -- and exists purely so
the log shows real out-of-sample error as training progresses. Hyperparameters are
not adjusted based on it.
"""

import math
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

tfkl = tf.keras.layers

# Paths resolve relative to this file, so the script runs from any directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))

# Fixed experimental design. These match the R benchmarks (Bohk-Ewald et al.
# 2018); changing any of them breaks comparability. Model hyperparameters
# (units, layers, learning rate, ...) live in train_ensemble.py instead.
AGE_MIN, AGE_MAX = 15, 44          # benchmark age1=15, age2=44
JUMP_OFF_YEARS = list(range(1985, 2014))   # every year 1985-2013, as in the benchmarks
FORECAST_LEN = 30                  # benchmark len=30
YEAR_MAX = 2015                    # last "known data" year; also the year-norm anchor
YEAR_MIN = 1950                    # year-norm anchor
EPOCHS = 50                        # fixed training budget; no early stopping
BATCH_SIZE = 256
PASSES_PER_EPOCH = 5               # data-sized draws per epoch (see train_and_forecast)
TOTAL_PASSES = EPOCHS * PASSES_PER_EPOCH   # = 250 per training phase
METHOD_NAME = "DL_NonLog"

# Which file supplies the rates. "square" = Lexis squares (completed age, ACY);
# "parallelogram" = vertical parallelograms (ARDY). Paper uses squares to match 
# most other benchmarks, but model can be trained with either
DATA_FILES = {
    "square": "asfr_1891_to_2024.txt",
    "parallelogram": "asfrVV_1891_to_2024.txt",
}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def normalize_year(year):
    """Map calendar year onto roughly [0, 1] via a fixed affine transform.

    Anchored to YEAR_MIN..YEAR_MAX rather than to each training set's own range,
    so training and forecasting always use the identical mapping. Forecast years
    run past YEAR_MAX and therefore map above 1.0 -- that is intended, the year
    input is a plain linear feature and extrapolates.
    """
    return (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)


def _get_row(data, max_val):
    """Draw one row of `data` at random and split it into (features, target).

    Rows are [geography, year, age, rate].
    """
    rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32)
    entry = data[rand_index, :]

    geography, year, age, rate = entry[0], entry[1], entry[2], entry[3]

    year = normalize_year(year)
    age = tf.cast(age, tf.int32)
    geography = tf.cast(geography, tf.int32)

    features = (tf.reshape(year, [1]), tf.reshape(age, [1]),
                tf.reshape(geography, [1]))
    return features, tf.reshape(rate, [1])


def prep_data(data):
    """Wrap a raw [geo, year, age, rate] array in a batched tf.data pipeline.

    Note how sampling works here: rows are drawn at random with replacement.
    """
    data = tf.cast(tf.convert_to_tensor(data), tf.float32)
    max_val = data.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices(np.arange(3000)).repeat()
    dataset = dataset.map(
        lambda _: _get_row(data, max_val=max_val),
        num_parallel_calls=4,
    )
    return dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


def make_features(geo, year, age):
    """Build the model's (year, age, geography) input tensors from raw columns.

    A plain deterministic conversion of every row, unlike prep_data's random
    stream -- used for validation (where a random sample would make the reported
    loss jitter epoch to epoch) and for the forecast grid.
    """
    return (
        tf.convert_to_tensor(normalize_year(year), dtype=tf.float32),
        tf.convert_to_tensor(age, dtype=tf.float32),
        tf.convert_to_tensor(geo, dtype=tf.float32),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def raw_to_variance(raw_var):
    """Turn the variance head's unconstrained output into a variance.

    Softplus enforces positivity, plus a small floor for numerical stability.
    Shared by the loss and by forecasting, so the variances written to disk are
    always the ones the head was trained to produce.
    """
    return tf.math.softplus(raw_var) + 1e-6


def gaussian_nll_loss(y_true, y_pred):
    """Gaussian negative log-likelihood, used to train the variance head.

    y_true is the observed rate, shape (batch, 1). y_pred is (batch, 2):
    column 0 is the predicted mean, column 1 the raw (pre-softplus) variance.

        loss = log(sigma^2) / 2 + (y - mu)^2 / (2 * sigma^2)

    The constant log(2*pi)/2 is dropped; it doesn't affect the gradients.
    """
    y_true = tf.reshape(y_true, [-1, 1])   # guard against (batch,) broadcasting
    mu = y_pred[:, 0:1]
    variance = raw_to_variance(y_pred[:, 1:2])

    nll = 0.5 * tf.math.log(variance) + 0.5 * tf.square(y_true - mu) / variance
    return tf.reduce_mean(nll)


def build_base_model(geo_dim, units, n_layers, dropout, age_embed_dim, geo_embed_dim,
                learning_rate):
    """Build and compile the ASFR network.

    Inputs are (normalized year, age, geography). Age and geography go through
    embedding layers; year enters as a scalar. The concatenated input vector
    passes through n_layers identical blocks (Dense -> LayerNormalization ->
    Dropout). Output is sigmoid-activated because rates live in [0, 1]. Loss is 
    MSE on the raw rate scale.

    Every hyperparameter is required -- there are no defaults, so a run's
    architecture is always fully determined by the calling script.
    """
    year = tfkl.Input(shape=(1,), dtype="float32", name="Year")
    age = tfkl.Input(shape=(1,), dtype="int32", name="Age")
    geography = tfkl.Input(shape=(1,), dtype="int32", name="Geography")

    # input_dim=55 covers ages 0-54; we only use 15-44, the rest sit unused.
    age_embed = tfkl.Flatten()(
        tfkl.Embedding(input_dim=55, output_dim=age_embed_dim, name="Age_embed")(age)
    )
    geo_embed = tfkl.Flatten()(
        tfkl.Embedding(input_dim=geo_dim, output_dim=geo_embed_dim,
                       name="Geography_embed")(geography)
    )

    x = tfkl.Concatenate()([year, age_embed, geo_embed])

    for _ in range(n_layers):
        x = tfkl.Dense(units, activation="relu")(x)
        x = tfkl.LayerNormalization()(x)
        x = tfkl.Dropout(dropout)(x)

    x = tfkl.Dense(1, activation="sigmoid", name="final")(x)

    model = tf.keras.Model(inputs=[year, age, geography], outputs=[x])
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def build_freeze_model(base_model):
    """Freeze a trained base model and add a variance head on top.

    The head is a single linear unit reading the last hidden layer (the input
    to "final"). The new model outputs [mu, raw_variance]; raw_to_variance turns
    the second column into a variance. Only the head is trainable, and it reuses
    the base model's learning rate.
    """
    for layer in base_model.layers:
        layer.trainable = False

    final = base_model.get_layer("final")
    mu = final.output
    raw_variance = tfkl.Dense(1, activation=None, name="raw_variance")(final.input)
    output = tfkl.Concatenate(name="output")([mu, raw_variance])

    freeze_model = tf.keras.Model(inputs=base_model.inputs, outputs=[output])
    learning_rate = float(base_model.optimizer.learning_rate)
    freeze_model.compile(loss=gaussian_nll_loss,
                         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return freeze_model


def train_freeze_ensemble_model(train_full, geo_dim, hparams, steps, epochs, val_data=None):
    """Train one ensemble member in two phases, `epochs` epochs each.

    Phase 1 (MSE): train the full network to predict the mean.
    Phase 2 (Gaussian NLL): freeze it, add a variance head, train only the head.

    There is no early stopping: the epoch count is a fixed budget (EPOCHS).

    val_data, if given, is the observed post-JOY window -- the same years the
    forecast is graded on. It is REPORTING ONLY. Nothing consumes val_loss: no
    early stopping, no LR schedule, no weight restoration, so passing it does not
    steer training and the forecast stays leak-free. (Checked empirically: with
    vs without val_data differs no more than two same-seed runs of the identical
    config do.) Its purpose is to let you watch true out-of-sample error fall (or
    not) epoch by epoch. Hand-tuning hyperparameters to minimise this val_loss
    would leak the test set back into the design.

    Returns (model, mean_history, var_history). model outputs [mu, raw_variance];
    mean_history losses are MSE, var_history losses are Gaussian NLL.
    """
    base_model = build_base_model(geo_dim, **hparams)
    mean_history = base_model.fit(train_full, steps_per_epoch=steps, epochs=epochs,
                                  validation_data=val_data, verbose=2)

    model = build_freeze_model(base_model)
    var_history = model.fit(train_full, steps_per_epoch=steps, epochs=epochs,
                            validation_data=val_data, verbose=2)

    tf.keras.backend.clear_session()
    return model, mean_history, var_history


# ---------------------------------------------------------------------------
# Data loading and country validation
# ---------------------------------------------------------------------------
# Whole-country series dropped in favour of their parts (DEUTE + DEUTW;
# GBRTENW + GBR_SCO + GBR_NIR), as in Bohk-Ewald et al. (2018), so the same
# cohorts are not trained on or scored twice. The R benchmarks drop the same
# codes (pnas_benchmarks/shared_helpers.R).
EXCLUDED_CODES = ["DEUTNP", "GBR_NP"]


def excluded_country_idx():
    """Numeric country indices of EXCLUDED_CODES, via geos_key.npy."""
    key = {str(code): int(idx)
           for code, idx in np.load(os.path.join(DATA_DIR, "geos_key.npy"), allow_pickle=True)}
    return [key[c] for c in EXCLUDED_CODES]


def load_data(lexis_shape="square", verbose=True):
    """Load the ASFR file for a Lexis shape, keep ages 15-44, drop EXCLUDED_CODES."""
    raw = np.loadtxt(os.path.join(DATA_DIR, DATA_FILES[lexis_shape]))
    asfr = raw[(raw[:, 2] >= AGE_MIN) & (raw[:, 2] <= AGE_MAX)]
    asfr = asfr[~np.isin(asfr[:, 0], excluded_country_idx())]

    if verbose:
        print(f"Data ({lexis_shape}): {DATA_FILES[lexis_shape]}")
        print(f"  raw {raw.shape} -> ages {AGE_MIN}-{AGE_MAX} {asfr.shape}")
        print(f"  years {int(asfr[:, 1].min())}-{int(asfr[:, 1].max())}, "
              f"countries {int(asfr[:, 0].max()) + 1}")
    return asfr


def validate_countries(asfr, verbose=True):
    """Find usable countries, and usable (country, JOY) pairs.

    Two filters, both mirroring the R benchmark so the comparison is like-for-like:

      1. Drop a country outright if it is missing any age in 15-44.
      2. For each JOY, drop a country if its series has a gap anywhere between
         its first observed year and that JOY, or if the JOY falls outside its
         data range (before it starts, or after YEAR_MAX).

    Returns (geo_dim, valid_countries, valid_combos) where valid_combos maps
    each JOY to the list of country indices usable at that JOY.
    """
    geo_dim = int(asfr[:, 0].max()) + 1
    required_ages = set(range(AGE_MIN, AGE_MAX + 1))

    # Cache each country's slice once; this loop is otherwise O(countries x JOYs)
    # full scans of the array.
    by_country = {}
    valid_countries = []
    country_min_year = {}

    for c in range(geo_dim):
        c_data = asfr[asfr[:, 0] == c]
        if len(c_data) == 0:
            continue
        missing = required_ages - set(c_data[:, 2].astype(int))
        if missing:
            if verbose:
                print(f"  skip country {c}: missing ages {sorted(missing)}")
            continue
        by_country[c] = c_data
        valid_countries.append(c)
        country_min_year[c] = int(c_data[:, 1].min())

    valid_combos = {}
    for joy in JUMP_OFF_YEARS:
        usable = []
        for c in valid_countries:
            min_yr = country_min_year[c]
            if joy > YEAR_MAX or joy < min_yr:
                continue
            available = set(by_country[c][:, 1].astype(int))
            gaps = set(range(min_yr, joy + 1)) - available
            if gaps:
                if verbose:
                    print(f"  skip country {c} @ JOY {joy}: missing years "
                          f"{sorted(gaps)[:5]}")
                continue
            usable.append(c)
        valid_combos[joy] = usable
        if verbose:
            print(f"JOY {joy}: {len(usable)} valid countries")

    if verbose:
        print(f"geo_dim={geo_dim}, valid countries={len(valid_countries)}")
    return geo_dim, valid_countries, valid_combos


# ---------------------------------------------------------------------------
# Training + forecasting
# ---------------------------------------------------------------------------
def train_and_forecast(asfr, geo_dim, valid_countries, valid_combos, hparams,
                       train_window, seed, models_dir=None, tag="run"):
    """Train one model per jump-off year and forecast JOY+1..JOY+FORECAST_LEN.

    hparams      dict of build_base_model's keyword arguments (all six required)
    train_window years of history to train on, counting back from each JOY.
                 None = use everything available.
    seed         seeds numpy/TF/python before each JOY. Note this makes runs
                 near-reproducible, not bit-identical: prep_data maps with
                 num_parallel_calls=4, so the order rows arrive in is not fully
                 determined by the seed. Measured residual between two runs at
                 the same seed is ~8e-5 per cell, versus ~2e-3 between different
                 seeds -- i.e. small enough not to affect config comparisons.
                 For exact reproducibility you would need
                 tf.config.experimental.enable_op_determinism() plus serial
                 mapping, at a meaningful speed cost.
    models_dir   where to save .keras files; None skips saving

    Returns a dict {joy: DataFrame of predicted *period* rates}, with the
    predicted mean in "Rate" and the predicted variance in "Var".
    """
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    predicted_rates = {}

    in_valid = np.isin(asfr[:, 0].astype(int), valid_countries)
    years = asfr[:, 1]

    for joy in JUMP_OFF_YEARS:
        print(f"\n{'=' * 60}\nJump-off year {joy}  (seed {seed})\n{'=' * 60}")

        # Training window lower bound: train on years in (after, joy].
        after = -np.inf if train_window is None else joy - train_window
        train_full = asfr[(years > after) & (years <= joy) & in_valid]

        print(f"  train_window={train_window} -> rows {train_full.shape[0]:,}")

        if train_full.shape[0] == 0:
            print("  no training data for this JOY -- skipping")
            continue

        # Validation = the observed rates over the forecast horizon, for exactly
        # the countries this JOY is graded on. Reported only, never trained or
        # selected on -- see train_freeze_ensemble_model. Years past the end of the data simply
        # aren't there, so later JOYs validate on a shorter window.
        in_graded = np.isin(asfr[:, 0].astype(int), valid_combos[joy])
        val_rows = asfr[(years > joy) & (years <= joy + FORECAST_LEN) & in_graded]
        val_data = None
        if val_rows.shape[0]:
            val_data = (make_features(val_rows[:, 0], val_rows[:, 1], val_rows[:, 2]),
                        tf.convert_to_tensor(val_rows[:, 3], dtype=tf.float32))
            print(f"  val (observed {int(val_rows[:, 1].min())}-"
                  f"{int(val_rows[:, 1].max())}): {val_rows.shape[0]:,} rows")
        else:
            print("  val: no observed data after this JOY -- training without it")

        # Because batches are randomly sampled rather than a real pass over the
        # data, scale "epoch length" to the split's size so an epoch means a
        # comparable number of examples regardless of how much history is in it.
        # Each epoch draws PASSES_PER_EPOCH x N rows, so each phase sees
        # TOTAL_PASSES x N rows in all.
        steps = math.ceil(train_full.shape[0] * PASSES_PER_EPOCH / BATCH_SIZE)
        print(f"  {EPOCHS} epochs x {steps} steps/epoch "
              f"(= {TOTAL_PASSES} passes over {train_full.shape[0]:,} rows per phase)")

        # Reseed per JOY so each jump-off year is independently reproducible.
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        model, mean_history, var_history = train_freeze_ensemble_model(
            prep_data(train_full), geo_dim, hparams, steps, EPOCHS, val_data)
        if val_data is not None:
            print(f"  final val_loss, mean phase (out-of-sample MSE): "
                  f"{mean_history.history['val_loss'][-1]:.6f}")
            print(f"  final val_loss, variance phase (out-of-sample NLL): "
                  f"{var_history.history['val_loss'][-1]:.6f}")

        # Forecast grid: every (valid country, future year, age) triple.
        forecast_years = np.arange(joy + 1, joy + FORECAST_LEN + 1)
        countries = sorted(valid_combos[joy])
        grid = np.array([(c, y, a)
                         for c in countries for y in forecast_years for a in ages])
        print(f"  forecast grid: {grid.shape[0]:,} cells "
              f"({len(countries)} countries x {len(forecast_years)} years x {len(ages)} ages)")

        features = make_features(grid[:, 0], grid[:, 1], grid[:, 2])
        # Model output is [mu, raw_variance] per cell.
        preds = model.predict(features)

        predicted_rates[joy] = pd.DataFrame({
            "Country": grid[:, 0].astype(int),
            "Year": grid[:, 1].astype(int),
            "Age": grid[:, 2].astype(int),
            "Rate": preds[:, 0],
            "Var": raw_to_variance(preds[:, 1]).numpy(),
        })

        if models_dir:
            os.makedirs(models_dir, exist_ok=True)
            path = os.path.join(models_dir, f"dl_cohort_{tag}_joy{joy}.keras")
            model.save(path)
            print(f"  saved {path}")

    print(f"\nAll jump-off years done ({len(predicted_rates)} trained).")
    return predicted_rates


# ---------------------------------------------------------------------------
# Period -> cohort conversion
# ---------------------------------------------------------------------------
def period_to_cohort(df):
    """Relabel rows by birth cohort: cohort = period year - age."""
    out = df.copy()
    out["Year"] = out["Year"] - out["Age"]
    return out


def _add_metadata(df, joy):
    df["JumpOffYear"] = joy
    df["Method"] = METHOD_NAME
    df["Key"] = df["Method"] + "_" + df["Country"].astype(str)
    return df


def build_predicted_cohort(predicted_rates):
    """Stack per-JOY predicted period rates into one cohort-indexed frame."""
    frames = [_add_metadata(period_to_cohort(predicted_rates[joy]), joy)
              for joy in JUMP_OFF_YEARS if joy in predicted_rates]
    out = pd.concat(frames, ignore_index=True)
    print(f"predicted cohort rates: {out.shape}")
    return out


def build_observed_cohort(asfr, valid_combos):
    """Observed cohort rates, one row per (Country, cohort, Age).

    An observed rate doesn't depend on the jump-off year, so there is no
    JumpOffYear column: each cell is written once, for every country usable at
    any JOY, and the eval notebooks match it to forecasts on (Country, Year, Age).
    Independent of the model, so this is identical across seeds -- the ensemble
    script builds it once.
    """
    all_obs = pd.DataFrame({
        "Country": asfr[:, 0].astype(int),
        "Year": asfr[:, 1].astype(int),
        "Age": asfr[:, 2].astype(int),
        "Rate": asfr[:, 3],
    })

    usable = set().union(*(valid_combos[joy] for joy in JUMP_OFF_YEARS))
    out = period_to_cohort(all_obs[all_obs["Country"].isin(usable)])
    out["Method"] = METHOD_NAME
    out["Key"] = out["Method"] + "_" + out["Country"].astype(str)
    out = out.sort_values(["Country", "Year", "Age"], ignore_index=True)
    print(f"observed cohort rates: {out.shape}")
    return out
