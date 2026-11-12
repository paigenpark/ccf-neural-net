# AGENTS.md — CCF-Net

Orientation for agents working in this repo. Read this first, then the docstrings
at the top of `pipeline.py` and `train_ensemble.py`, which are detailed and accurate.

## What this project is

Research code for the paper **"Can a Simple Neural Network Improve Cohort Fertility
Forecasts?"** (Paige Park, UC Berkeley Demography). The draft is in
`manuscript/ccf-forecasting-11-9.pdf`.

The paper asks whether a small, untuned Multi-Layer Perceptron (**CCF-Net**) can forecast
**completed cohort fertility (CCF)** better than the established demographic methods. It
re-runs the top methods from **Bohk-Ewald, Li & Myrskylä (2018, PNAS)** using their code
and their scoring rules, then adds CCF-Net to the comparison.

Headline findings:
- CCF-Net ranks 1st overall across jump-off years 1985–2013. It ties Myrskylä et al. on
  the KS dominance test and wins on mean APE (3.02% vs 3.14%). It also has the lowest
  CCF RMSE.
- Its forecasts are higher than the benchmarks', which was more accurate in the
  historical windows (especially post-socialist Eastern Europe).
- Two weaknesses: its error swings a lot from one jump-off year to the next (poor for JOY
  2006–2011), and from a 2019 jump-off it forecasts a very low CCF (~1.4) that may not be
  realistic.

## Domain vocabulary

| Term | Meaning |
|---|---|
| ASFR | Age-specific fertility rate (births per woman at a given age and year). |
| CCF / CF | Completed cohort fertility: sum of a birth cohort's ASFRs over ages 15–40. |
| JOY | Jump-off year: the last observed year a model may see. Everything after is forecast. |
| Truncation age | A cohort's age in the JOY. Ages ≤ it are observed, ages above are forecast. Scoring uses truncation ages 20–39. |
| Lexis square (ACY) | Period rates by calendar year × completed age. `asfrRR`, the main input. |
| Vertical parallelogram (ARDY) | Rates by age reached during the year × cohort. `asfrVV`, used only by Schmertmann. |
| Period → cohort | `cohort = year − age`. Every method forecasts period rates and then converts. |
| HFD | Human Fertility Database, the only data source. |

## Experimental design (fixed — do not treat as tuning knobs)

- Rolling origin: one forecast per JOY, for **every year 1985–2013**.
- Each forecast covers JOY+1 … JOY+30 (30 years), ages 15–44. Scoring uses ages 15–40.
- Countries are included only if they have every age 15–44 and no missing years from
  their first year up to the JOY. Methods with a fixed observation window (Lee and de
  Beer: 40 years; Ševčíková: 30) only need that window to be complete.
- **DEUTNP and GBR_NP are excluded** (their sub-regions are kept), so no cohort is counted
  twice. Python and R drop them the same way.
- Evaluation follows Bohk-Ewald et al.: methods are ranked by a pairwise one-sided
  two-sample KS test on APE of CCF, with p-values from a **country-level paired
  permutation test** (10,000 permutations). Ties are broken by mean APE. The tables also
  report APE thresholds at 50/80/90/95% and an RMSE column, which is the root mean square
  of the CCF APEs (in %).
- **CCF-Net is deliberately not tuned.** Its hyperparameters were set before any
  evaluation, and there is no validation split. The validation loss printed during
  training is the graded test period and is shown **for reporting only**. Never use it to
  pick hyperparameters; that would leak the test set. The paper's honesty claim depends
  on this.

Changing any constant in the "Fixed experimental design" block of `pipeline.py`
(`AGE_MIN/MAX`, `JUMP_OFF_YEARS`, `FORECAST_LEN`, `YEAR_MIN/MAX`, `EPOCHS`, …) makes the
results impossible to compare with the R benchmarks.

## CCF-Net architecture (paper config `u110_l4_d0.0_w40`)

- Inputs: year (scalar, min-max scaled with a fixed 1950–2015 anchor; forecast years
  extrapolate above 1.0), age (embedding of size 5), and country (embedding of size 5).
- 4 hidden blocks: Dense(110, ReLU) → LayerNorm → Dropout(0.0). Output is Dense(1, sigmoid).
- **Two-phase "freeze" training** in each ensemble member:
  1. Mean phase: the full network is trained on MSE.
  2. Variance phase: the network is frozen, a single linear unit on the last hidden layer
     (softplus + 1e-6) is added, and only that unit is trained on Gaussian NLL.
- Training uses the 40 years up to each JOY, Adam with lr 1e-3, batch size 256, and 50
  epochs per phase. Each epoch draws 5×N rows **at random with replacement** (see
  `prep_data`), so an "epoch" here is not one pass over the data. There is no early
  stopping.
- Ensemble: 5 seeds (0–4), combined as an equal-weight Gaussian mixture.
  Rate = mean of the member means. Var = mean of the member variances (aleatoric) +
  variance of the member means (epistemic, ddof=0). The 95% interval is ±1.96·SD,
  clipped to [0, 1].
- Same-seed runs agree to about 8e-5 per cell (not bit-identical, because of parallel
  `tf.data` mapping). Different seeds differ by about 2e-3.

## Benchmarks (R, in `code/pnas_benchmarks/`)

Adapted from the Bohk-Ewald et al. reference code (GPL).

| Dir | Method | Notes |
|---|---|---|
| `FreezeRates/` | Freeze rates | Naive baseline. |
| `Lee1993Log/` | Lee (1993) | Lee–Carter on log ASFR (`demography::lca`), 40-yr window. `lca.log.r` is a helper. |
| `Myrskyla2013/` | Myrskylä et al. (2013) | Extrapolates the 5-yr trend for 5 years, then freezes it. |
| `deBeer1985and1989/` | de Beer | ARIMA(1,0,0) over age × (2,1,0) over cohort, `minpack.lm`. Japan JOY 1986–90 blow up and are dropped from the common pool. |
| `Schmertmann2013/` | Schmertmann et al. (2014) | Bayesian, on `asfrVV` + `exposVV`. Starts at age 16 and stops at JOY+29. |
| `Sevcikova2016/` | Ševčíková et al. (2016) | UN `bayesTFR` MCMC, 30-yr window. Like Schmertmann, starts at age 16 and stops at JOY+29. **Takes hours.** Smoke-test via env vars, e.g. `SEV_ITER=60 SEV_CHAINS=1 SEV_JOYS=1985`. |

`shared_helpers.R` holds the country exclusion and `collapse_obs()` (one observed row per
cell). Each script has a `joys <- c(...)` line (`JOYS` in Sevcikova) and writes `<m>_forecasts_cohort.csv` and
`<m>_obs_cohort.csv` to `data/`; some also write `*_uncertainty.csv`. The scripts use
`here()`, so run them from the repo root.

## Repo layout

```
code/
  data_preparation/
    split_period_data.py # raw asfrRR.txt -> asfr_1891_to_2024.txt + geos_key.npy
    split_vv_data.py     # raw asfrVV.txt -> asfrVV_1891_to_2024.txt (run second)
  dl_training_with_uncertainty/
    pipeline.py          # all model/data machinery; no config lives here
    train_ensemble.py    # CONFIG block (hyperparams, VARIANT, SEEDS) + ensembling
  pnas_benchmarks/       # six R benchmark models (see above)
  evaluation/
    evaluation_functions.py         # small legacy MSE helpers
    eval_figures_cohort.ipynb       # main results: Table 1, Figs 2–5, Table A3, Fig A2
    eval_figures_variants.ipynb     # hyperparameter-variant sensitivity (Table A2)
    eval_figures_uncertainty.ipynb  # 95% interval coverage (Appendix C)
    eval_figures_future.ipynb       # 2019 jump-off forecasts (Fig 6)
  future_forecasts/
    run_all.py           # re-runs every model at JOY 2019 (patched R copies go to data/future_forecasts/_patched/)
    run_dl_future.py     # DL-only future run; overrides train_ensemble globals
    verify_future.py     # audits future outputs; --reproduce checks the patcher at JOY 2010
diagram/ccfnet_architecture.py   # draws Figure 1 (pdf/svg/png); edit its CONFIG block
manuscript/                      # paper PDF
data/                            # gitignored; created locally — inputs and outputs live here
```

## Data (not in the repo)

HFD data needs registration, so it is not committed (`data/` is gitignored). Download
the raw files from the HFD into `data/asfr/` and `data/exposure/`, then run
`code/data_preparation/split_period_data.py` followed by `split_vv_data.py`. Files the
code expects in `data/`:
- `asfr_1891_to_2024.txt`: numeric `[country_idx, year, age, rate]` from `asfrRR` (squares)
- `asfrVV_1891_to_2024.txt`: the same layout from `asfrVV` (parallelograms)
- `geos_key.npy`: mapping from country code to numeric index (order of first appearance in `asfrRR.txt`)
- `asfr/asfrRR.txt`, `asfr/asfrVV.txt`, `exposure/exposVV.txt`: raw HFD files, used by the
  prep scripts and the R scripts

The prep scripts drop non-numeric ages and ages above 54, and cap rates at 1.
If the data is missing, ask the user for it before trying to run anything.

Outputs: `data/dl_forecasts_<VARIANT>*.csv`, `dl_obs_cohort_<VARIANT>.csv`,
`dl_config_<VARIANT>.json`, the R `*_forecasts_cohort.csv` files,
`data/sensitivity_checks/` (DL variants), and `data/future_forecasts/`.

**CSV schema (all methods):** `Country, Year, Age, Rate, JumpOffYear, Method, Key`. Here
`Year` is the **birth cohort**, not the calendar year (period year = Year + Age). Observed
files have no `JumpOffYear` and are matched on `(Country, Year, Age)`.

## Running things

Environment: Python ≥3.13 managed by `uv` (`pyproject.toml`, `uv.lock`; the package is
named `deep-fert`). R needs `here`, `dplyr`, `demography`, `forecast`, `ftsa`,
`minpack.lm`, `bayesTFR`, `svd`, `MASS`, and `strucchange`.

```bash
uv sync
# Train the ensemble (slow: 29 JOYs × 5 seeds × 2 phases). Trim SEEDS while iterating.
nohup .venv/bin/python code/dl_training_with_uncertainty/train_ensemble.py > train_<VARIANT>.log 2>&1 &
Rscript code/pnas_benchmarks/FreezeRates/FreezeRates.R
.venv/bin/python code/future_forecasts/run_all.py            # 5 R models + DL at JOY 2019
.venv/bin/python code/future_forecasts/verify_future.py [--reproduce]
```

Use `.venv/bin/python`. A bare `python` won't find TensorFlow. `train_ensemble.py`
ignores SIGTTOU/SIGTTIN so it can run in the background.

To add a DL run to the figures, give it a new `VARIANT` name and add an entry to
`MODEL_REGISTRY` in `eval_figures_cohort.ipynb`, which is the single source of truth for
every table and plot. Notebooks read `../../data/<name>.csv`.

## Canonical CCF-Net outputs

The paper's CCF-Net is `VARIANT = "CCF_Net_u110_l4_d0.0_w40"` in `train_ensemble.py`,
written to `data/dl_forecasts_CCF_Net_u110_l4_d0.0_w40.csv` (plus `dl_obs_cohort_…` and
`…_uncertainty.csv`). The cohort, uncertainty and variants notebooks and
`verify_future.py` all read this run. The hyperparameter variants for Table A2 are in
`data/sensitivity_checks/`. Figures are saved to `figures/` at the repo root.

## Guidelines for agents

- Keep the comparison fair. Do not change the shared design constants, the country
  filters, or the scoring age window unless the user asks, and say so whenever a change
  affects comparability with Bohk-Ewald et al.
- Never tune on the post-JOY validation loss, and never add early stopping or LR
  schedules driven by it.
- Hyperparameters go in the `train_ensemble.py` CONFIG block. `pipeline.py` takes
  everything as explicit arguments.
- Give every new configuration a new `VARIANT` name, so earlier outputs are never
  overwritten.
- Don't edit the R benchmarks or the DL trainer to make future forecasts.
  `future_forecasts/` re-points them without modifying them, and `verify_future.py
  --reproduce` checks that this is faithful.
- Training and Sevcikova are long jobs. Ask before launching them, and use small seed,
  JOY, or iteration counts for smoke tests.
- The manuscript is the reference for what the code is meant to do. If the code and the
  paper disagree, point it out instead of quietly choosing one.
