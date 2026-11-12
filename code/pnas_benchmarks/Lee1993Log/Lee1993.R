#######################################################################
#######################################################################
## 'validate-forecast-methods' is a program that provides the R source code
## used for the calculations in the project: 
## Bohk-Ewald, Christina, Peng Li, and Mikko Myrskylä (2017). 
## Assessing the accuracy of cohort fertility forecasts. 
## Presented in session: Statistical methods in demography 
## at the PAA 2017 Annual Meeting, Chicago, IL, USA, April 27-April 29, 2017. 
## (c) Copyright 2018, Christina Bohk-Ewald, Peng Li, Mikko Myrskylä

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program (see LICENSE.txt in the root of this source code).
## If not, see <http://www.gnu.org/licenses/>.
#######################################################################
#######################################################################

###############################################################################
#    Project: Cohort fertility forecastes and their accuracy
#    CBE, PL and MM
#    2016.09.23
#    
#    Method: Lee-Carter method provided in demography package (Hyndman)
#     
#    Function.1: Create demogdata object from raw data matrices
#    Method14_Lee1993Log.R
############################################################
#### Input file: 
#     ASFR: age-specific fertility rate
#     EXPOS: age-specific exposures
#     
#### Parameters
#      joy: jump of year
#      obs: length of observation
#      age1 : star age
#      age2 : end age
#      parameter : adjust, jumpchoice, and prediction method
#             1. adjust: c("dt", "dxt", "none","popconst") 
#                        "popconst":set the populations in every age groups are the same, and used dxt method to adjust.
#             2. jumpchoice: c("fit", "actual")
#             3. pred method: c("rw", "arma"): predict using random walk or arma mehod
#      len : length of forecasting period
#      pop : population

#### Output: 
#      obsASFR: observed ASFR, age * period
#      obsCASFR: observed ASFR, age * cohort
#      predASFR: forecasted ASFR, age * period
#      predASFRlowerPI[80,90] : lower boundary of PI
#      predASFRupperPI[80/90] : upper boundary of PI
#      predCASFR: forecasted ASFR, age * cohort
#      predCASFRlowerPI[80,90] : lower boundary of PI
#      predCASFRupperPI[80/90] : upper boundary of PI
#      method: name of the method
#      parameter: parameter of the method
#      year: starting year of observed period, JOY and last year of forecasting
#      age: range of age
#      obs: number of years of observation
#      len: forecasting length
#      cohort: c1: oldest cohort at the starting year of observation
#              c2: youngest cohort at the starting year of observation
#              c3: oldest cohort at JOY
#              c4: youngest cohort at JOY
#
#### NOTE:
#      
#### Ref:
# 1. Lee RD (1993) Modeling and forecasting the time series of US fertility: Age distribution, range, 
#    and ultimate level. International Journal of Forecasting 9:187-202.
#    
###############################################################################
# vector of required packages
required_packages <- c("demography", "tidyverse", "glue", "here")

# function to check and install missing packages 
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# apply the function to all required packages
invisible(lapply(required_packages, install_if_missing))

# load the libraries
lapply(required_packages, library, character.only = TRUE)
require(svd)

source(here("code/pnas_benchmarks/Lee1993Log/lca.log.r"))

###########################################
Method14_Lee1993Log.R <- function(ASFR,
                                  EXPOS,
                                  joy = 1985, 
                                  obs = 30,
                                  age1 = 15, 
                                  age2 = 44,
                                  parameter = c("dt","actual","Boundary","arma"),
                                  len = 30,
                                  pop=""
){
  ################ Period to cohort function
  asfr_period_to_cohort <- function(input){
  temp <- expand.grid(as.numeric(dimnames(input)[[1]]),as.numeric(dimnames(input)[[2]]))
  cohort <- temp[,2] - temp[,1]
  output <- tapply(as.vector(input), list(temp[,1], cohort),sum)
  return(as.data.frame(output))
}
  ################ Data step
  if(is.na(joy))
    joy <- max(as.numeric(colnames(ASFR)[!is.na(apply(ASFR,2,sum))]))
  ########
  year1 <- joy - obs + 1
  year2 <- joy
  year3 <- year2 + len
  ########
  
  # extract input data: exposure
  expos.raw <- EXPOS[paste(age1:age2),paste(year1:year2)]
  # extract input data: asfr
  fert.raw <- ASFR[paste(age1:age2),paste(year1:year2)]
  fert.raw[fert.raw<=0.00001] <- 0.00001
  raw.data <- matrix(NA,nrow=nrow(fert.raw),ncol=(year3-year1+1))
  colnames(raw.data) <- paste(year1 + c(1:ncol(raw.data))-1)
  rownames(raw.data) <- rownames(fert.raw)
  raw.data[,colnames(fert.raw)] <- fert.raw
  
  ########
  if(any(is.na(sum(raw.data[,paste(c(year1,year2))]))) | !any(apply(raw.data,2,function(x) all(!is.na(x)))) )
    return(list(pop=pop,
                obsASFR=ASFR,
                obsCASFR=asfr_period_to_cohort(ASFR),
                predASFR=NA,
                predCASFR=NA,
                predCASFRlowerPI80 = NA,
                predASFRlowerPI80  = NA,
                predCASFRupperPI80 = NA,
                predASFRupperPI80  = NA, 
                predCASFRlowerPI95 = NA,
                predASFRlowerPI95  = NA, 
                predCASFRupperPI95 = NA,
                predASFRupperPI95  = NA,
                method="Lee (1993)",
                parameter=parameter,
                label=ifelse(all(!is.na(parameter)),paste("LeeLog",paste(parameter,collapse = "_"),sep="_"),"LeeLog"),
                year=c(year1,year2,year3),
                age=c(age1,age2),
                obs=obs,
                len=len,
                cohort=c((year1-age2),(year1-age1),(year2-age2),(year2-age1))))
  
  ################ Model step
  #### Fit model based on Lee-Carter method
  if(parameter[1] == "popconst"){
    parameter[1] = "dxt"
    pop <-  matrix(100000,nrow = nrow(expos.raw), ncol= ncol(expos.raw))
    dimnames(pop) <- dimnames(expos.raw)
    expos.raw <- pop
  }
  fert.fit <- lca(fert.raw, expos.raw, adjust=parameter[1],restype="rates",label=pop)
  fert.fcast1 <- forecast.lca(fert.fit,h=year3-year2,level=80,method=parameter[3],jumpchoice=parameter[2])
  fert.fcast2 <- forecast.lca(fert.fit,h=year3-year2,level=95,method=parameter[3],jumpchoice=parameter[2])
  
  data <- cbind(fert.raw,fert.fcast1$rate$total)
  colnames(data) <- paste(c(as.numeric(colnames(fert.raw)),fert.fcast1$year))
  dataupper80 <- cbind(fert.raw,fert.fcast1$rate$upper)
  colnames(dataupper80) <- colnames(data)
  datalower80 <- cbind(fert.raw,fert.fcast1$rate$lower)
  colnames(datalower80) <- colnames(data)
  
  dataupper95 <- cbind(fert.raw,fert.fcast2$rate$upper)
  colnames(dataupper95) <- colnames(data)
  datalower95 <- cbind(fert.raw,fert.fcast2$rate$lower)
  colnames(datalower95) <- colnames(data)
  ####
  ################ Output step
  predASFR <- data
  # Cap forecast rates at 1.0: the log-bilinear model can extrapolate k(t)
  # into exponential blow-up (rates >> 1, which are impossible for ASFR).
  # This matches the rate cap applied in the DL/observed pipeline.
  predASFR[predASFR > 1] <- 1
  predCASFR <- asfr_period_to_cohort(predASFR)
  
  predASFRlowerPI80 <- datalower80 
  predASFRupperPI80 <- dataupper80
  predCASFRlowerPI80 <- asfr_period_to_cohort(predASFRlowerPI80)
  predCASFRupperPI80 <- asfr_period_to_cohort(predASFRupperPI80)
  
  predASFRlowerPI95 <- datalower95 
  predASFRupperPI95 <- dataupper95
  predCASFRlowerPI95 <- asfr_period_to_cohort(predASFRlowerPI95)
  predCASFRupperPI95 <- asfr_period_to_cohort(predASFRupperPI95)
  
  ################ Output step
  return(list(pop=pop,
              obsASFR=ASFR,
              obsCASFR=asfr_period_to_cohort(ASFR),
              predASFR=predASFR,
              predCASFR=predCASFR,
              predCASFRlowerPI80 = predCASFRlowerPI80,
              predASFRlowerPI80  = predASFRlowerPI80,
              predCASFRupperPI80 = predCASFRupperPI80,
              predASFRupperPI80  = predASFRupperPI80, 
              predCASFRlowerPI95 = predCASFRlowerPI95,
              predASFRlowerPI95  = predASFRlowerPI95, 
              predCASFRupperPI95 = predCASFRupperPI95,
              predASFRupperPI95  = predASFRupperPI95,
              method="Lee (1993)",
              parameter=parameter,
              label=ifelse(all(!is.na(parameter)),paste("LeeLog",paste(parameter,collapse = "_"),sep="_"),"LeeLog"),
              year=c(year1,year2,year3),
              age=c(age1,age2),
              obs=obs,
              len=len,
              cohort=c((year1-age2),(year1-age1),(year2-age2),(year2-age1))))
}

###########################################
#### Prep ASFR data 
# load and prepare data
path <- here("data")
asfr_training <- read.table(paste(path, "asfr_1891_to_2024.txt", sep = "/"), 
                              header = FALSE)
# Drop the whole-country DEU / GBR series; their parts stay (see shared_helpers.R)
source(here("code/pnas_benchmarks/shared_helpers.R"))
asfr_training <- asfr_training[!(asfr_training[, 1] %in% excluded_country_idx(path)), ]
countries <- unique(asfr_training[,1])
ages <- unique(asfr_training[,3])
# forecasted_years <- 2006:2015
colnames(asfr_training) <- c('Country', 'Year', 'Age', 'Rate')


all_forecasts <- list()
all_forecasts_pi <- list()
all_obs <- list()

##############################
# ROBUST LOOP WITH GAP CHECKS
##############################

joys <- c(1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013)

# Helper: convert a cohort matrix (age x cohort) to long-format data frame
cohort_matrix_to_long <- function(mat, country, joy, method, key) {
  if (is.null(mat) || all(is.na(mat))) return(NULL)
  ages <- as.numeric(rownames(mat))
  cohorts <- as.numeric(colnames(mat))
  expand <- expand.grid(Age = ages, Year = cohorts)
  expand$Rate <- as.vector(as.matrix(mat))
  expand$Country <- country
  expand$JumpOffYear <- joy
  expand$Method <- method
  expand$Key <- key
  expand <- expand[!is.na(expand$Rate), c("Country", "Year", "Age", "Rate", "JumpOffYear", "Method", "Key")]
  return(expand)
}

# Helper: like cohort_matrix_to_long, but for the forecast plus its 80% and 95%
# prediction interval bounds, one column each. All five matrices share the same
# (age x cohort) layout, so they are flattened in the same order.
# forecast.lca's "lower"/"upper" are the rates at the lower/upper bound of k(t),
# so they swap wherever bx < 0; pmin/pmax puts them back in order. Bounds are
# clipped to [0, 1], matching the cap on the forecast itself. They are intervals
# on individual age-specific rates and reflect k(t) forecast uncertainty only.
cohort_pi_to_long <- function(item, country, joy, method, key) {
  mat <- item$predCASFR
  if (is.null(mat) || all(is.na(mat))) return(NULL)
  ages <- as.numeric(rownames(mat))
  cohorts <- as.numeric(colnames(mat))
  expand <- expand.grid(Age = ages, Year = cohorts)
  expand$Rate <- as.vector(as.matrix(mat))
  for (lvl in c("80", "95")) {
    a <- as.vector(as.matrix(item[[paste0("predCASFRlowerPI", lvl)]]))
    b <- as.vector(as.matrix(item[[paste0("predCASFRupperPI", lvl)]]))
    expand[[paste0("Lower", lvl)]] <- pmax(pmin(a, b), 0)
    expand[[paste0("Upper", lvl)]] <- pmin(pmax(a, b), 1)
  }
  expand$Country <- country
  expand$JumpOffYear <- joy
  expand$Method <- method
  expand$Key <- key
  expand <- expand[!is.na(expand$Rate), c("Country", "Year", "Age", "Rate",
                                          "Lower80", "Upper80", "Lower95", "Upper95",
                                          "JumpOffYear", "Method", "Key")]
  return(expand)
}

for (i in countries) {
  # 1. Filter and Prepare Data
  filtered <- asfr_training |> filter(Country == i)

  if(nrow(filtered) == 0) next

  fx_wide <- filtered |>
    pivot_wider(names_from = Year, values_from = Rate) |>
    arrange(Age)

  ASFR_mat <- as.matrix(fx_wide |> select(-Country, -Age))
  rownames(ASFR_mat) <- as.character(sort(unique(filtered$Age)))
  colnames(ASFR_mat) <- as.character(sort(unique(filtered$Year)))

  # Create dummy exposure matrix
  EXPOS_mat <- matrix(100000, nrow = nrow(ASFR_mat), ncol = ncol(ASFR_mat))
  dimnames(EXPOS_mat) <- dimnames(ASFR_mat)

  # 2. GAP CHECK: AGES
  # The function strictly requires ages 15 through 44.
  # We must check that EVERY single age in 15:44 exists in the data.
  required_ages <- as.character(15:44)
  missing_ages <- setdiff(required_ages, rownames(ASFR_mat))

  if (length(missing_ages) > 0) {
    message(paste("Skipping Country", i, "- Missing specific ages:", paste(head(missing_ages), collapse=",")))
    next
  }

  available_years <- as.numeric(colnames(ASFR_mat))
  min_data_year <- min(available_years)

  # Lee variant uses the best-performing observation window from the paper: 40 years.
  obs_window <- 40

  # 3. Iterate through Jump-Off Years
  for (joy in joys) {

    # Check bounds: joy must exist and there must be a full 40-year window of history.
    window_start <- joy - obs_window + 1
    if (joy > max(available_years) || window_start < min_data_year) {
      next
    }

    # 4. GAP CHECK: YEARS
    required_years_seq <- window_start:joy
    missing_years <- setdiff(as.character(required_years_seq), colnames(ASFR_mat))

    if (length(missing_years) > 0) {
      message(paste("Skipping", i, joy, "- Gap in data. Missing years:", paste(head(missing_years), collapse=",")))
      next
    }

    # If we pass all checks, use the fixed 40-year window and run
    obs_years <- obs_window

    tryCatch({
      result <- Method14_Lee1993Log.R(
        ASFR  = ASFR_mat,
        EXPOS = EXPOS_mat,
        joy   = joy,
        obs   = obs_years,
        age1  = 15,
        age2  = 44,
        parameter = c("none", "actual", "arma"),
        len   = 30,
        pop   = i
      )

      method_name <- "LeeCarter_Log"
      key_name <- paste0("LeeCarter_Log_", i)

      # Extract forecasted cohort ASFR
      fc <- cohort_matrix_to_long(result$predCASFR, i, joy, method_name, key_name)
      if (!is.null(fc)) all_forecasts[[length(all_forecasts) + 1]] <- fc

      # Same forecast with its prediction interval bounds
      fc_pi <- cohort_pi_to_long(result, i, joy, method_name, key_name)
      if (!is.null(fc_pi)) all_forecasts_pi[[length(all_forecasts_pi) + 1]] <- fc_pi

      # Extract observed cohort ASFR
      ob <- cohort_matrix_to_long(result$obsCASFR, i, joy, method_name, key_name)
      if (!is.null(ob)) all_obs[[length(all_obs) + 1]] <- ob

    }, error = function(e) {
      message(paste("Error in Country", i, "Year", joy, ":", e$message))
    })
  }
}

# Combine and save as CSVs
forecasts_df <- bind_rows(all_forecasts)
obs_df <- collapse_obs(all_obs)   # one row per cell, no JumpOffYear (shared_helpers.R)

write.csv(forecasts_df, file = file.path(path, "lee2_forecasts_cohort.csv"), row.names = FALSE)
write.csv(obs_df, file = file.path(path, "lee2_obs_cohort.csv"), row.names = FALSE)

message(paste("Saved", nrow(forecasts_df), "forecast rows to", file.path(path, "lee2_forecasts_cohort.csv")))
message(paste("Saved", nrow(obs_df), "observed rows to", file.path(path, "lee2_obs_cohort.csv")))

# Forecasts with prediction intervals go to their own file, leaving the two
# above untouched.
forecasts_pi_df <- bind_rows(all_forecasts_pi)

write.csv(forecasts_pi_df, file = file.path(path, "lee2_forecasts_cohort_uncertainty.csv"), row.names = FALSE)

message(paste("Saved", nrow(forecasts_pi_df), "forecast rows with intervals to", file.path(path, "lee2_forecasts_cohort_uncertainty.csv")))
