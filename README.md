# CCF-Net

Replication code for **"Can a Simple Neural Network Improve Cohort Fertility Forecasts?"**
(Paige Park, UC Berkeley Demography). The draft is in `manuscript/`.

The paper compares a small neural network (CCF-Net) with six established cohort fertility
forecasting methods from Bohk-Ewald, Li & Myrskylä (2018, PNAS). Every method forecasts
from each jump-off year 1985–2013 and is scored according to the reference paper's criteria.

## 1. Get the data

All data comes from the [Human Fertility Database](https://www.humanfertility.org) (HFD),
which needs a free registration, so it is not included here. The paper uses data
downloaded in **June 2026**. HFD revises its series over time, so a later download may give
slightly different numbers.

Download these files and put them here:

```
data/asfr/asfrRR.txt         # ASFR by calendar year and age (Lexis squares)
data/asfr/asfrVV.txt         # ASFR by cohort and age (vertical parallelograms)
data/exposure/exposVV.txt    # exposures, vertical parallelograms
```

## 2. Set up the environment

Python (3.13+, managed with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

Always run Python with `.venv/bin/python`. A bare `python` won't find TensorFlow.

R, with these packages:

```r
install.packages(c("here", "dplyr", "demography", "forecast", "ftsa", "minpack.lm",
                   "bayesTFR", "svd", "MASS", "strucchange"))
```

Run every command below from the repo root.

## 3. Prepare the data

```bash
.venv/bin/python code/data_preparation/split_period_data.py
.venv/bin/python code/data_preparation/split_vv_data.py      # must run second
```

## 4. Run the benchmark models (R)

```bash
Rscript code/pnas_benchmarks/FreezeRates/FreezeRates.R
Rscript code/pnas_benchmarks/Lee1993Log/Lee1993.R
Rscript code/pnas_benchmarks/Myrskyla2013/Myrskyla2013.R
Rscript code/pnas_benchmarks/deBeer1985and1989/deBeer1985and1989.R
Rscript code/pnas_benchmarks/Schmertmann2013/Schmertmann2013.R
Rscript code/pnas_benchmarks/Sevcikova2016/Sevcikova2016.r   # slow: takes hours
```

Each script writes its forecasts to `data/`. To check that Ševčíková runs before
starting the full job, try a short run:

```bash
SEV_ITER=60 SEV_CHAINS=1 SEV_JOYS=1985 Rscript code/pnas_benchmarks/Sevcikova2016/Sevcikova2016.r
```

## 5. Train CCF-Net

```bash
nohup .venv/bin/python code/dl_training_with_uncertainty/train_ensemble.py \
    > train_CCF_Net_u110_l4_d0.0_w40.log 2>&1 &
```

This is slow: it trains 5 seeds for each of the 29 jump-off years. The settings are already
set to the paper's model (`CCF_Net_u110_l4_d0.0_w40`). Output goes to
`data/dl_forecasts_CCF_Net_u110_l4_d0.0_w40*.csv`.

**Sensitivity variants (Table A2).** To reproduce these, edit the CONFIG block at the top of
`train_ensemble.py`, set `OUT_SUBDIR = "sensitivity_checks"`, and rerun once for each
variant. Each run changes one setting from the paper's model:

| VARIANT | Change |
|---|---|
| `u256_l4_d0.0_w40` | `UNITS = 256` |
| `u110_l5_d0.0_w40` | `N_LAYERS = 5` |
| `u110_l4_d0.1_w40` | `DROPOUT = 0.1` |
| `u110_l4_d0.0_w35` | `TRAIN_WINDOW = 35` |
| `u110_l4_d0.0_wNone` | `TRAIN_WINDOW = None` (all history) |

## 6. Future forecasts (jump-off year 2019)

```bash
.venv/bin/python code/future_forecasts/run_all.py           # 5 R models + CCF-Net
.venv/bin/python code/future_forecasts/run_all.py --all     # also Ševčíková (slow)
.venv/bin/python code/future_forecasts/verify_future.py     # optional check
```

This reruns the models from 2019 without editing them. Output goes to
`data/future_forecasts/`.

## 7. Make the tables and figures

Open the notebooks in `code/evaluation/` and run them top to bottom:

| Notebook | Produces |
|---|---|
| `eval_figures_cohort.ipynb` | Main results: Table 1, Figures 2–5, Table A3, Figure A2 |
| `eval_figures_variants.ipynb` | Sensitivity variants (Table A2) |
| `eval_figures_uncertainty.ipynb` | Interval coverage (Appendix C) |
| `eval_figures_future.ipynb` | 2019 forecasts (Figure 6) |

Figures are saved to `figures/`. Figure 1 (the architecture diagram) comes from
`diagram/ccfnet_architecture.py`.

## Notes

- **Results won't match to the last digit.** Runs with the same seed differ by about 8e-5 per
  rate because TensorFlow processes data in parallel.
- **CCF-Net was not tuned.** The "validation loss" printed during training is computed on the
  test period and is shown for information only.
- `AGENTS.md` has more detail on the experimental design and code layout.

## Citation

Park, Paige. 2026. "Can a Simple Neural Network Improve Cohort Fertility Forecasts?"
Working paper, Department of Demography, University of California, Berkeley.

## License

GNU General Public License v3.0 (see `LICENSE`). The R benchmark code is adapted from the
Bohk-Ewald, Li & Myrskylä (2018) reference code, which is also GPL v3.
