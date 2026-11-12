###############################################################################
#  UN Bayesian approach (Alkema et al. 2011; Sevcikova et al. 2016)
#  Cohort-fertility benchmark, as used in Bohk-Ewald et al. (2018, PNAS).
#
#  Best-performing variant only: SpecificPattern-EqualWeights (SpPEW),
#  30 years of observation.
#
#  Two-step procedure (Sevcikova et al. 2016 = Alkema 2011 TFR + age profiles):
#   1. TFR forecast via the UN Bayesian Hierarchical Model (R package bayesTFR),
#      fit on the HFD pool (NOT the ~200 UN countries), single years of age &
#      calendar time, 60,000 MCMC iterations. Point forecast = posterior median.
#   2. Split each forecast-year TFR over age with a proportional PASFR profile
#      (weighted mean of (a) convergence to a global pattern and (b) linear
#      extrapolation of the national trend; global pattern = SpecificPattern +
#      EqualWeights) to get age-specific PERIOD rates TFR(t)*PASFR(a,t). Each
#      cohort is then completed by summing these rates along its diagonal
#      (age a of cohort c takes year c+a); the cohort total falls out of its
#      own rates -- there is NO period-TFR -> completed-CF (MAB) mapping.
#
#  NB: the period-TFR -> CF mapping via p = round(c + MABbar) (Cheng & Lin; also
#  the *Alkema 2011* TFR-only variant, SI Table S29 ~11% median APE) was REMOVED
#  2026-07-24. It forced every cohort to a single year's period TFR, capping
#  accuracy at ~10%; the age-specific
#  step 2 is what makes this the rank-4 Sevcikova 2016 method (~1.2% median APE).
#
#  Output (shared cohort-benchmark contract):
#    data/sevcikova_forecasts_cohort.csv
#    data/sevcikova_obs_cohort.csv
#  columns: Country, Year(=cohort), Age, Rate, JumpOffYear, Method, Key
#  cohort = Year - Age; Rate = non-cumulative age-specific cohort fertility;
#  Key = "Sevcikova2016_<country>".
#
#  Forecasts with prediction intervals go to their own file,
#    data/sevcikova_forecasts_cohort_uncertainty.csv
#  with the same rows and Rate plus SD, Lower80, Upper80, Lower95, Upper95
#  (same columns as schmertmann2013_forecasts_cohort_uncertainty.csv). The
#  intervals carry the bayesTFR posterior of TFR(t) only: the PASFR age profile
#  is a deterministic point estimate, so each cell's bounds are
#  PASFR(a,t) * quantile(TFR(t)). They are intervals on individual age-specific
#  rates; completed cohort fertility needs the joint TFR trajectories.
#
#  ---------------------------------------------------------------------------
#  Age-profile step follows Sevcikova et al. (2016), Sect. 15.3.2 (the
#  "Convergence Method"), implemented in logit-PASFR space:
#    - global-pattern projection      p^I_t   (eq. 15.14)
#    - national-trend continuation    p^II_t  (eq. 15.15)
#    - blend by convergence fraction  p_t     (eq. 15.16)
#    - convergence timing tg via Phase-III / ultimate TFR (eqs. 15.18-15.19)
#  Global model pattern = SpPEW: per-COI Specific reference set (ranked >= COI),
#  Equal Weights, averaging recent PASFRs (footnote 6 procedure, but with
#  Bohk-Ewald's formal SpPEW ranking replacing the subjective country list).
#
#  Documented adaptations to Bohk-Ewald's single-year setting (differ from the
#  chapter's 7 five-year age groups / 5-year periods):
#   * single years of age & calendar time throughout;
#   * national trend uses two smoothed endpoints (BASE_SPAN-year averages at tr
#     and tr-NT_SPAN) rather than one 5-year period each;
#   * cohort ASFR is read off the period-PASFR diagonal: age a of cohort c takes
#     year c+a. Each missing age gets rate TFR(c+a)*PASFR(a,c+a) directly; the
#     completed cohort total is the sum of observed ages plus these forecast
#     rates (no CF target, no renormalization).
#   * t_P3 is proxied by the observed TFR trough; the Sect. 15.3.2.5 "frozen
#     late-childbearing" exception is not applied (second-order for these JOYs).
###############################################################################

## ------------------------------ setup -------------------------------------
required_packages <- c("tidyverse", "here", "bayesTFR")
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
invisible(lapply(required_packages, install_if_missing))
suppressMessages(invisible(lapply(required_packages, library, character.only = TRUE)))

path <- here("data")

## ---------------------------- configuration -------------------------------
AGE1 <- 15                 # first modeled age
AGE2 <- 44                 # last modeled age
OBS  <- 30                 # years of observation (SpPEW best variant)
LEN  <- 30                 # forecast horizon (shared contract)
JOYS <- c(1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013)
{ jenv <- Sys.getenv("SEV_JOYS"); if (nzchar(jenv))
    JOYS <- as.integer(strsplit(jenv, ",")[[1]]) }

# bayesTFR MCMC settings. Paper: ~60,000 iterations are sufficient.
# Split across chains for mixing; total post-burnin kept for prediction.
# Override for quick smoke tests, e.g. SEV_ITER=60 SEV_CHAINS=1 SEV_JOYS=1985.
env_int <- function(name, default) {
  v <- Sys.getenv(name); if (nzchar(v)) as.integer(v) else default
}
MCMC_CHAINS  <- env_int("SEV_CHAINS", 3)
MCMC_ITER    <- env_int("SEV_ITER", 20000)   # per chain -> 60,000 total
MCMC_BURNIN  <- env_int("SEV_BURNIN", 2000)  # per chain (estimation)
PRED_BURNIN  <- env_int("SEV_PREDBURN", min(2000, MCMC_ITER %/% 2))
PRED_NRTRAJ  <- env_int("SEV_NRTRAJ", 2000)

# Prediction-interval columns -> bayesTFR posterior quantile of TFR (80% / 95%).
PI_QUANTILES <- c(Lower80 = "0.1", Upper80 = "0.9", Lower95 = "0.025", Upper95 = "0.975")
PI_COLS      <- c("SD", names(PI_QUANTILES))

# SpPEW ranking weights (paper): 20% dTFR, 70% dMAB, 10% phase-3 indicator.
W_TFR <- 0.20
W_MAB <- 0.70
W_P3  <- 0.10
TOP_FRACTION <- 0.20       # global pattern = top 20% of the pool
MIN_PATTERN  <- 2          # minimum countries to define a pattern
PHASE3_TFR   <- 2.1        # below-replacement threshold for phase-3 indicator

# Convergence method (Sevcikova et al. 2016, Sect. 15.3.2), adapted to single
# years of age / calendar time.
#   BASE_SPAN: years averaged for the base-period PASFR pr (emulates one 5-yr
#              period; also used for the global pattern's "recent" profiles).
#   NT_SPAN:   T in eq. 15.15 -- span (years) of the national-trend
#              extrapolation. Chapter uses 3 five-year periods = 15 years.
#   TG_MIN_LAG: lower bound tr+10 on the global-pattern arrival year (eq. 15.19,
#              "two 5-year periods").
BASE_SPAN  <- 5
NT_SPAN    <- 15
TG_MIN_LAG <- 10

# MCMC outputs are cached here (data/ is gitignored). Delete to force a rerun.
MCMC_ROOT <- { r <- Sys.getenv("SEV_MCMC_ROOT")
  if (nzchar(r)) r else file.path(path, "sevcikova_mcmc") }
dir.create(MCMC_ROOT, showWarnings = FALSE, recursive = TRUE)

## ------------------------------ load data ---------------------------------
asfr <- read.table(file.path(path, "asfr_1891_to_2024.txt"), header = FALSE)
colnames(asfr) <- c("Country", "Year", "Age", "Rate")
asfr <- asfr |> filter(Age >= AGE1, Age <= AGE2)
# Drop the whole-country DEU / GBR series; their parts stay (see shared_helpers.R)
source(here("code/pnas_benchmarks/shared_helpers.R"))
asfr <- asfr |> filter(!Country %in% excluded_country_idx(path))

ages_full <- AGE1:AGE2
n_age     <- length(ages_full)

## --------------------------- helper functions -----------------------------

# Per country-year summaries over the modeled age range.
#   TFR = sum of single-year ASFR; MAB = mean age of childbearing (age+0.5).
# Returns only country-years with the complete age range present.
country_year_summaries <- function(df) {
  df |>
    group_by(Country, Year) |>
    summarise(TFR = sum(Rate),
              MAB = sum((Age + 0.5) * Rate) / sum(Rate),
              nage = n(), .groups = "drop") |>
    filter(nage == n_age) |>
    select(-nage)
}

# Wide observed ASFR matrix (age x year) for one country.
country_asfr_matrix <- function(df, country) {
  sub <- df |> filter(Country == country)
  yrs <- sort(unique(sub$Year))
  m <- matrix(NA_real_, nrow = n_age, ncol = length(yrs),
              dimnames = list(as.character(ages_full), as.character(yrs)))
  m[cbind(match(sub$Age, ages_full), match(sub$Year, yrs))] <- sub$Rate
  m
}

# Proportional age profile (sums to 1) from a set of yearly ASFR columns,
# averaged over the supplied years. `mat` is age x year with year colnames.
proportional_profile <- function(mat, years) {
  cols <- intersect(as.character(years), colnames(mat))
  if (length(cols) == 0) return(NULL)
  avg <- rowMeans(mat[, cols, drop = FALSE], na.rm = TRUE)
  if (all(!is.finite(avg)) || sum(avg, na.rm = TRUE) <= 0) return(NULL)
  avg[!is.finite(avg)] <- 0
  avg / sum(avg)
}

# logit / expit with a floor so PASFR zeros at extreme ages stay finite.
logit_p <- function(p, eps = 1e-6) { p <- pmin(pmax(p, eps), 1 - eps); log(p / (1 - p)) }
expit    <- function(x) 1 / (1 + exp(-x))

# Renormalize a non-negative age vector to sum to 1 (NULL if degenerate).
renorm <- function(p) {
  p[!is.finite(p)] <- 0; p[p < 0] <- 0
  s <- sum(p); if (s <= 0) NULL else p / s
}

# Cohort matrix (age x cohort) -> long data frame in the shared output contract.
cohort_matrix_to_long <- function(mat, country, joy, method, key) {
  if (is.null(mat) || all(is.na(mat))) return(NULL)
  ages    <- as.numeric(rownames(mat))
  cohorts <- as.numeric(colnames(mat))
  ex <- expand.grid(Age = ages, Year = cohorts)
  ex$Rate <- as.vector(as.matrix(mat))
  ex$Country <- country
  ex$JumpOffYear <- joy
  ex$Method <- method
  ex$Key <- key
  ex[!is.na(ex$Rate), c("Country", "Year", "Age", "Rate", "JumpOffYear", "Method", "Key")]
}

# Like cohort_matrix_to_long, but with the forecast's SD and 80% / 95% interval
# bounds, one column each. `pi_mats` is a list of age x cohort matrices named by
# PI_COLS, aligned with `mat`. Bounds are clipped to [0, 1].
cohort_pi_to_long <- function(mat, pi_mats, country, joy, method, key) {
  if (is.null(mat) || all(is.na(mat))) return(NULL)
  ages    <- as.numeric(rownames(mat))
  cohorts <- as.numeric(colnames(mat))
  ex <- expand.grid(Age = ages, Year = cohorts)
  ex$Rate <- as.vector(as.matrix(mat))
  for (s in PI_COLS) ex[[s]] <- as.vector(pi_mats[[s]])
  ex$Lower80 <- pmax(ex$Lower80, 0); ex$Upper80 <- pmin(ex$Upper80, 1)
  ex$Lower95 <- pmax(ex$Lower95, 0); ex$Upper95 <- pmin(ex$Upper95, 1)
  ex$Country <- country
  ex$JumpOffYear <- joy
  ex$Method <- method
  ex$Key <- key
  ex[!is.na(ex$Rate), c("Country", "Year", "Age", "Rate", PI_COLS,
                        "JumpOffYear", "Method", "Key")]
}

## ------------------- step 1: TFR forecast via bayesTFR ---------------------
# Fit the UN BHM on the HFD pool for one JOY (cached) and return a matrix of
# median TFR by country x forecast-year (JOY+1 .. JOY+LEN). Observed TFR for
# years <= JOY is taken directly from the data by the caller.
run_bayestfr_for_joy <- function(joy, pool_summ) {
  start_year <- joy - OBS + 1
  out_dir    <- file.path(MCMC_ROOT, paste0("joy_", joy))

  win <- pool_summ |> filter(Year >= start_year, Year <= joy)
  # keep countries with the full OBS-year window
  full <- win |> count(Country) |> filter(n == OBS)
  win  <- win |> semi_join(full, by = "Country")
  countries <- sort(unique(win$Country))

  wide <- win |> select(Country, Year, TFR) |>
    pivot_wider(names_from = Year, values_from = TFR) |> arrange(Country)
  cc <- wide$Country + 1L               # bayesTFR country codes (avoid 0)

  tfr_file <- data.frame(country = paste0("C", wide$Country),
                         country_code = cc, include_code = 2L,
                         check.names = FALSE)
  tfr_file <- cbind(tfr_file, wide |> select(-Country))
  loc_file <- data.frame(name = paste0("C", wide$Country), country_code = cc,
                         reg_code = 900L, reg_name = "World",
                         area_code = 900L, area_name = "World",
                         location_type = 4L, tree_level = 4L, check.names = FALSE)

  # input files live OUTSIDE output.dir (run.tfr.mcmc requires an empty out dir)
  tfr_path <- file.path(MCMC_ROOT, paste0("tfr_", joy, ".txt"))
  loc_path <- file.path(MCMC_ROOT, paste0("loc_", joy, ".txt"))

  pred_dir <- file.path(out_dir, "predictions")
  if (!dir.exists(pred_dir)) {
    unlink(out_dir, recursive = TRUE)
    write.table(tfr_file, tfr_path, sep = "\t", row.names = FALSE, quote = FALSE)
    write.table(loc_file, loc_path, sep = "\t", row.names = FALSE, quote = FALSE)

    message(sprintf("[JOY %d] running bayesTFR MCMC (%d chains x %d iter) on %d countries...",
                    joy, MCMC_CHAINS, MCMC_ITER, length(countries)))
    m <- run.tfr.mcmc(nr.chains = MCMC_CHAINS, iter = MCMC_ITER, thin = 1,
                      annual = TRUE, use.wpp.data = FALSE,
                      start.year = start_year, present.year = joy,
                      my.tfr.file = tfr_path, my.locations.file = loc_path,
                      output.dir = out_dir, seed = 1, verbose = FALSE)
    # Phase-III (post-transition recovery) BHM. The SI applies the TFR model to
    # phase-2 AND phase-3 countries; estimating tfr3 (vs use.tfr3=FALSE's fixed
    # AR(1) mu=2.1) lets below-replacement countries follow their own recovery
    # instead of all reverting to replacement.
    run.tfr3.mcmc(sim.dir = out_dir, nr.chains = MCMC_CHAINS, iter = MCMC_ITER,
                  thin = 1, seed = 1, verbose = FALSE)
    tfr.predict(m, end.year = joy + LEN, burnin = PRED_BURNIN,
                nr.traj = PRED_NRTRAJ, use.tfr3 = TRUE, burnin3 = PRED_BURNIN,
                save.as.ascii = 0, output.dir = out_dir, verbose = FALSE)
  } else {
    message(sprintf("[JOY %d] using cached MCMC in %s", joy, out_dir))
  }

  pred <- get.tfr.prediction(sim.dir = out_dir)
  fyears <- (joy + 1):(joy + LEN)
  med <- matrix(NA_real_, nrow = length(countries), ncol = length(fyears),
                dimnames = list(as.character(countries), as.character(fyears)))
  # posterior spread of TFR, same shape as `med`: one matrix per PI_COLS entry
  spread <- setNames(rep(list(med), length(PI_COLS)), PI_COLS)
  for (i in seq_along(countries)) {
    ri  <- as.character(countries[i])
    tab <- tfr.trajectories.table(pred, country = cc[i], pi = c(80, 95))
    rows <- intersect(as.character(fyears), rownames(tab))
    med[ri, rows] <- tab[rows, "median"]
    for (s in names(PI_QUANTILES)) spread[[s]][ri, rows] <- tab[rows, PI_QUANTILES[[s]]]
    # SD has no quantile-table column; take it over the saved trajectories
    traj <- get.tfr.trajectories(pred, cc[i])            # year x trajectory
    rows <- intersect(as.character(fyears), rownames(traj))
    spread$SD[ri, rows] <- apply(traj[rows, , drop = FALSE], 1, sd)
  }
  list(median = med, spread = spread, countries = countries, start_year = start_year)
}

## --------- step 2: SpPEW global pattern + cohort age profiles --------------

# Rank the pool for one JOY by the SpPEW weighting function and return per-country
# score plus the top-20% set. dTFR favors LOW TFR; dMAB favors HIGH MAB.
rank_pool <- function(joy, pool_summ) {
  at_joy <- pool_summ |> filter(Year == joy)
  # phase-3 indicator: below replacement at JOY
  at_joy <- at_joy |> mutate(p3 = as.integer(TFR < PHASE3_TFR))
  tfr_min <- min(at_joy$TFR); tfr_max <- max(at_joy$TFR)
  mab_min <- min(at_joy$MAB); mab_max <- max(at_joy$MAB)
  at_joy <- at_joy |>
    mutate(dTFR = if (tfr_max > tfr_min) (tfr_max - TFR) / (tfr_max - tfr_min) else 0,
           dMAB = if (mab_max > mab_min) (MAB - mab_min) / (mab_max - mab_min) else 0,
           score = W_TFR * dTFR + W_MAB * dMAB + W_P3 * p3) |>
    arrange(desc(score))
  n_top <- max(MIN_PATTERN, ceiling(TOP_FRACTION * nrow(at_joy)))
  at_joy$top20 <- FALSE
  at_joy$top20[seq_len(min(n_top, nrow(at_joy)))] <- TRUE
  at_joy
}

# Global-pattern proportional profile for a COI (SpecificPattern + EqualWeights).
# Specific: reference countries = those in the top-20% ranked at least as high as
# the COI (COI may be included). Equal weights: simple mean of their profiles.
global_pattern_profile <- function(coi, ranked, joy, asfr_mats) {
  coi_score <- ranked$score[ranked$Country == coi]
  refs <- ranked |> filter(top20, score >= coi_score) |> pull(Country)
  if (length(refs) < MIN_PATTERN) refs <- head(ranked$Country[ranked$top20], MIN_PATTERN)
  profs <- lapply(refs, function(rc) {
    m <- asfr_mats[[as.character(rc)]]
    if (is.null(m)) return(NULL)
    proportional_profile(m, (joy - BASE_SPAN + 1):joy)   # recent PASFRs
  })
  profs <- profs[!vapply(profs, is.null, logical(1))]
  if (length(profs) == 0) return(NULL)
  renorm(rowMeans(do.call(cbind, profs)))                # equal weights
}

# Estimate tg, the calendar year the COI reaches the global pattern
# (Sect. 15.3.2.4, Case 1). tfr_traj = named TFR by year (observed + forecast
# median); tr = JOY (base period); te = end of forecast horizon (~ultimate).
estimate_tg <- function(tfr_traj, tr, te) {
  yrs <- as.numeric(names(tfr_traj))
  obs <- tfr_traj[yrs <= tr]
  # start of Phase III ~ trough of TFR in observation (entry into recovery)
  t_p3 <- as.numeric(names(obs)[which.min(obs)])
  fu   <- tfr_traj[as.character(te)]              # ultimate level ~ TFR at te
  if (!is.finite(fu)) return(min(tr + TG_MIN_LAG, te))
  cand <- yrs[yrs > t_p3 & tfr_traj >= fu]        # eq. 15.18
  tu   <- if (length(cand)) min(cand) else te
  min(max(tu, tr + TG_MIN_LAG), te)               # eq. 15.19
}

# Period PASFR (age x future-year) via the convergence method (eqs 15.14-15.16).
# pr, p_start, pg are proportional age profiles (sum to 1) over ages_full;
# pr = base (tr), p_start = base minus NT_SPAN (t_{r-T}), pg = global pattern.
period_pasfr_matrix <- function(pr, p_start, pg, tr, tg, years) {
  lpr <- logit_p(pr); lpg <- logit_p(pg); lps <- logit_p(p_start)
  out <- matrix(NA_real_, nrow = length(pr), ncol = length(years),
                dimnames = list(names(pr), as.character(years)))
  for (j in seq_along(years)) {
    t <- years[j]
    s <- if (tg > tr) min(max((t - tr) / (tg - tr), 0), 1) else 1
    p_i  <- renorm(expit(lpr + s * (lpg - lpr)))                     # 15.14
    p_ii <- renorm(expit(lpr + ((t - tr) / NT_SPAN) * (lpr - lps)))  # 15.15
    if (is.null(p_i) || is.null(p_ii)) next
    p_t  <- renorm(expit(s * logit_p(p_i) + (1 - s) * logit_p(p_ii)))# 15.16
    if (!is.null(p_t)) out[, j] <- p_t
  }
  out
}

## ------------------------------- run --------------------------------------
pool_summ_all <- country_year_summaries(asfr)

all_forecasts <- list()
all_forecasts_pi <- list()
all_obs <- list()

for (joy in JOYS) {
  message(sprintf("==================== JOY %d ====================", joy))

  # --- step 1: TFR forecast (median) ---
  tfr_fc <- run_bayestfr_for_joy(joy, pool_summ_all)
  countries <- tfr_fc$countries

  # observed TFR/MAB lookups within the pool for this JOY
  pool_summ <- pool_summ_all |> filter(Country %in% countries)
  ranked <- rank_pool(joy, pool_summ)

  # cache each country's observed ASFR matrix (age x year)
  asfr_mats <- setNames(lapply(countries, function(ct) country_asfr_matrix(asfr, ct)),
                        as.character(countries))

  # TFR lookup: observed for year <= JOY, forecast median for year > JOY
  obs_tfr <- pool_summ |> select(Country, Year, TFR)

  for (ct in countries) {
    key   <- paste0("Sevcikova2016_", ct)
    m_obs <- asfr_mats[[as.character(ct)]]
    obs_years <- as.numeric(colnames(m_obs))
    start_year <- tfr_fc$start_year

    # completing cohorts: started (reached AGE1) but not complete (not yet AGE2) at JOY
    cohorts <- (joy - (AGE2 - 1)):(joy - AGE1)

    # per-country TFR getter (observed or forecast median)
    tfr_at <- function(p) {
      if (p <= joy) {
        v <- obs_tfr$TFR[obs_tfr$Country == ct & obs_tfr$Year == p]
        if (length(v) == 1) return(v) else return(NA_real_)
      }
      row <- as.character(ct); col <- as.character(p)
      if (row %in% rownames(tfr_fc$median) && col %in% colnames(tfr_fc$median))
        return(tfr_fc$median[row, col])
      NA_real_
    }

    # --- convergence-method period PASFR surface for this country (once) ---
    fyears  <- (joy + 1):(joy + LEN)
    pr      <- proportional_profile(m_obs, (joy - BASE_SPAN + 1):joy)
    p_start <- proportional_profile(
      m_obs, (joy - NT_SPAN - BASE_SPAN + 1):(joy - NT_SPAN))
    pg      <- global_pattern_profile(ct, ranked, joy, asfr_mats)
    if (is.null(pr) || is.null(p_start) || is.null(pg)) next

    tfr_traj <- setNames(vapply(start_year:(joy + LEN), tfr_at, numeric(1)),
                         as.character(start_year:(joy + LEN)))
    tg    <- estimate_tg(tfr_traj, joy, joy + LEN)
    pasfr <- period_pasfr_matrix(pr, p_start, pg, joy, tg, fyears)  # age x year

    pred_cols <- list()
    pi_cols   <- setNames(rep(list(list()), length(PI_COLS)), PI_COLS)
    for (c in cohorts) {
      last_obs_age <- joy - c                       # oldest observed age of cohort c
      fc_ages <- (last_obs_age + 1):AGE2            # ages still to forecast
      if (length(fc_ages) == 0) next

      # require a clean observed base so the notebook can reconstruct the cohort
      # trajectory (observed history + these forecast ages) up to completion.
      obs_ages <- AGE1:last_obs_age
      obs_vals <- vapply(obs_ages, function(a) {
        yr <- as.character(c + a)
        if (yr %in% colnames(m_obs)) m_obs[as.character(a), yr] else NA_real_
      }, numeric(1))
      if (any(!is.finite(obs_vals))) next

      # Sevcikova (2016) forecasts age-SPECIFIC period rates (TFR x PASFR); the
      # cohort is completed by reading these off the diagonal (age a of cohort c
      # takes year c+a) and letting the cohort total fall out of its own rates --
      # no period-TFR->CF (MAB) mapping, which is the Alkema (2011) TFR-only step.
      fc_rate <- vapply(fc_ages, function(a) {
        yr <- c + a
        pv <- pasfr[as.character(a), as.character(yr)]
        tv <- tfr_at(yr)
        if (!is.finite(pv) || !is.finite(tv)) NA_real_ else pv * tv
      }, numeric(1))
      if (any(!is.finite(fc_rate))) next

      col <- setNames(rep(NA_real_, n_age), as.character(ages_full))
      col[as.character(fc_ages)] <- fc_rate
      pred_cols[[as.character(c)]] <- col

      # Same cells with the TFR posterior spread in place of the median. PASFR
      # is a fixed positive scalar per cell, so it scales TFR's SD and quantiles
      # directly.
      fc_years <- as.character(c + fc_ages)
      pv <- pasfr[cbind(as.character(fc_ages), fc_years)]
      for (s in PI_COLS) {
        col_s <- setNames(rep(NA_real_, n_age), as.character(ages_full))
        col_s[as.character(fc_ages)] <- pv * tfr_fc$spread[[s]][as.character(ct), fc_years]
        pi_cols[[s]][[as.character(c)]] <- col_s
      }
    }

    if (length(pred_cols) == 0) next
    predCASFR <- do.call(cbind, pred_cols)
    colnames(predCASFR) <- names(pred_cols)
    rownames(predCASFR) <- as.character(ages_full)

    fc <- cohort_matrix_to_long(predCASFR, ct, joy, "Sevcikova2016", key)
    if (!is.null(fc)) all_forecasts[[length(all_forecasts) + 1]] <- fc

    pi_mats <- lapply(pi_cols, function(x) do.call(cbind, x))   # same forecast + intervals
    fc_pi <- cohort_pi_to_long(predCASFR, pi_mats, ct, joy, "Sevcikova2016", key)
    if (!is.null(fc_pi)) all_forecasts_pi[[length(all_forecasts_pi) + 1]] <- fc_pi

    # observed cohort truth (age x cohort), same period-to-cohort convention
    obsCASFR <- {
      temp <- expand.grid(as.numeric(rownames(m_obs)), as.numeric(colnames(m_obs)))
      coh  <- temp[, 2] - temp[, 1]
      as.data.frame(tapply(as.vector(m_obs), list(temp[, 1], coh), sum))
    }
    ob <- cohort_matrix_to_long(obsCASFR, ct, joy, "Sevcikova2016", key)
    if (!is.null(ob)) all_obs[[length(all_obs) + 1]] <- ob
  }
}

## ------------------------------ write out ---------------------------------
forecasts_df <- bind_rows(all_forecasts)
obs_df       <- collapse_obs(all_obs)   # one row per cell, no JumpOffYear (shared_helpers.R)

write.csv(forecasts_df, file = file.path(path, "sevcikova_forecasts_cohort.csv"),
          row.names = FALSE)
write.csv(obs_df, file = file.path(path, "sevcikova_obs_cohort.csv"),
          row.names = FALSE)

message(sprintf("Saved %d forecast rows to %s",
                nrow(forecasts_df), file.path(path, "sevcikova_forecasts_cohort.csv")))
message(sprintf("Saved %d observed rows to %s",
                nrow(obs_df), file.path(path, "sevcikova_obs_cohort.csv")))

# Forecasts with prediction intervals go to their own file, leaving the two above
# untouched. Observed rates have no interval, so they are scored against
# sevcikova_obs_cohort.csv as before.
forecasts_pi_df <- bind_rows(all_forecasts_pi)

write.csv(forecasts_pi_df,
          file = file.path(path, "sevcikova_forecasts_cohort_uncertainty.csv"),
          row.names = FALSE)

message(sprintf("Saved %d forecast rows with intervals to %s", nrow(forecasts_pi_df),
                file.path(path, "sevcikova_forecasts_cohort_uncertainty.csv")))
