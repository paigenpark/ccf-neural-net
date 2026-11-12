## Helpers shared by every cohort benchmark script.

## -----------------------------------------------------------------------------
## Countries left out of every cohort benchmark.
##
## The HFD carries whole-country series for Germany (DEUTNP) and the UK (GBR_NP)
## alongside their constituents (DEUTE + DEUTW; GBRTENW + GBR_SCO + GBR_NIR).
## Bohk-Ewald, Li & Myrskyla (2018, SI p.16) drop the totals and keep the parts,
## so the same cohorts are not scored twice. We do the same.
##
## The benchmarks key countries by a 0-based numeric index = order of first
## appearance in asfr/asfrRR.txt (what split_period_data.py writes to
## geos_key.npy), so the codes are mapped through that order here.
## -----------------------------------------------------------------------------
EXCLUDED_CODES <- c("DEUTNP", "GBR_NP")

excluded_country_idx <- function(path) {
  rr_codes <- read.table(file.path(path, "asfr", "asfrRR.txt"),
                         skip = 3, header = FALSE,
                         colClasses = c("character", "NULL", "NULL", "NULL"))[[1]]
  code_levels <- unique(rr_codes)
  idx <- match(EXCLUDED_CODES, code_levels) - 1L
  stopifnot(!anyNA(idx))
  idx
}

## -----------------------------------------------------------------------------
## Observed cohort rates, written once.
##
## The scripts collect the observed frame inside the jump-off-year loop, but an
## observed rate does not depend on the jump-off year, so writing it per JOY
## just repeats the same cells (x29 with every JOY 1985-2013). This stacks the
## per-JOY frames, checks that every copy of a cell agrees, and keeps one row per
## (Country, Year, Age) with no JumpOffYear column. Cells seen at any JOY are
## kept, so the result is the union of what each JOY observed.
## -----------------------------------------------------------------------------
collapse_obs <- function(all_obs) {
  obs <- dplyr::bind_rows(all_obs)
  spread <- obs |>
    dplyr::group_by(Country, Year, Age) |>
    dplyr::summarise(d = max(Rate) - min(Rate), .groups = "drop")
  stopifnot(max(spread$d) < 1e-12)   # identical across JOYs, or collapsing loses data
  obs |>
    dplyr::distinct(Country, Year, Age, .keep_all = TRUE) |>
    dplyr::select(-JumpOffYear) |>
    dplyr::arrange(Country, Year, Age)
}
