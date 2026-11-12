#######################################################################
#######################################################################
## 'validate-forecast-methods' is a program that provides the R source code
## used for the calculations in the project:
## Bohk-Ewald, Christina, Peng Li, and Mikko Myrskyla (2017).
## Assessing the accuracy of cohort fertility forecasts.
## Presented in session: Statistical methods in demography
## at the PAA 2017 Annual Meeting, Chicago, IL, USA, April 27-April 29, 2017.
## (c) Copyright 2018, Christina Bohk-Ewald, Peng Li, Mikko Myrskyla

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
#    2017.02.22
#
#    Method:
#
#
#    Method18_Myrskyla2013.R
############################################################
#### Input file:
#     asfr_period_hfd: age-specific fertility rate, a 3D array (age * period * country)
#
#
#### Parameters
#      joy: jump of year (first_forecast_year)
#      obs: length of observation
#      age1 : star age
#      age2 : end age
#      parameter : parameter of the method
#            1. [years_to_extrapolate]
#      len : length of forecasting period
#      pop : population
#### Output:
#      obsASFR: observed ASFR, age * period
#      obsCASFR: observed ASFR, age * cohort
#      predASFR: forecasted ASFR, age * period
#      predCASFR: forecasted ASFR, age * cohort
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
#### NOTE:
#      This method loops over all countries internally so that the prediction
#      interval variance is pooled across countries, as in the original PNAS
#      "validate-forecast-methods" package. The point forecasts (predCASFR) do
#      not depend on that pooled variance, so the cohort completions remain
#      directly comparable to the Lee (1993) and Freeze Rates benchmarks.
#### Ref:
#   1. Myrskyla, M., Goldstein, J.R. and Cheng, Y.H.A., 2013. New cohort fertility forecasts for the developed world:
#      rises, falls, and reversals. Population and Development Review, 39(1), pp.31-56.
#
###############################################################################
########################################################################################################
# function takes a matrix of fertility rates (ages as rows, years as columns)
Myrskyla2013_function <- function(data){
  n.col <- ncol(data)
  a <- data[,n.col] # extracts rates for last year 
  b <- (data[,n.col] - data[,1])/(n.col-1) # calculates slope 
                                            # between first and last year of training window
                                            # function is only fed data from 5 years prior to JOY
  kt <- apply(data-a,2,function(x){sum(x*b)/sum(b^2)})

  sigma <- kt[n.col]-kt[1] / (n.col-1)
  epsilon <- kt[2:n.col]-kt[2:n.col-1] - sigma
  return(list(a=a,b=b,kt=kt,sigma=sigma,epsilon=epsilon))
}
########################################################################################################
Method18_Myrskyla2013.R <- function(asfr_period_hfd,
                          joy = 2014,
                          obs = 5,
                          age1 = 15,
                          age2 = 44,
                          parameter = c(5),
                          len = 30
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
    joy <- max(as.numeric(dimnames(asfr_period_hfd)[[2]]))

  Country <- dimnames(asfr_period_hfd)[[3]]
  ########
  year1 <- joy - obs + 1
  year2 <- joy
  year3 <- year2 + len

  ################ Model step
  raw.data <- array(NA,dim=c(length(age1:age2),length(year1:year3),length(Country)),dimnames=list(paste(age1:age2),paste(year1:year3),Country))
  raw.datalowerPI80 <- array(NA,dim=c(length(age1:age2),length(year1:year3),length(Country)),dimnames=list(paste(age1:age2),paste(year1:year3),Country))
  raw.datalowerPI95 <- array(NA,dim=c(length(age1:age2),length(year1:year3),length(Country)),dimnames=list(paste(age1:age2),paste(year1:year3),Country))
  raw.dataupperPI80 <- array(NA,dim=c(length(age1:age2),length(year1:year3),length(Country)),dimnames=list(paste(age1:age2),paste(year1:year3),Country))
  raw.dataupperPI95 <- array(NA,dim=c(length(age1:age2),length(year1:year3),length(Country)),dimnames=list(paste(age1:age2),paste(year1:year3),Country))
  var.data <- array(0, dim=c(length(age1:age2),length(year1:year3),length(Country)),dimnames=list(paste(age1:age2),paste(year1:year3),Country))

  ######## fitt model
  obj_all <- list()
  for(ct in Country){
    fert.raw <- asfr_period_hfd[paste(age1:age2),paste(year1:year2),ct]
    raw.data[paste(age1:age2),paste(year1:year2),ct] <- fert.raw
    raw.datalowerPI80[paste(age1:age2),paste(year1:year2),ct] <- fert.raw
    raw.datalowerPI95[paste(age1:age2),paste(year1:year2),ct] <- fert.raw
    raw.dataupperPI80[paste(age1:age2),paste(year1:year2),ct] <- fert.raw
    raw.dataupperPI95[paste(age1:age2),paste(year1:year2),ct] <- fert.raw
    obj <- Myrskyla2013_function(fert.raw[,paste((year2-4):year2)])
    obj_all[[ct]] <- obj
  }

  ######## estimate variance
  epsilon_Variance <- var(do.call("c",lapply(obj_all, "[[", "epsilon")),na.rm=T)
  sigma_Variance = epsilon_Variance / (5 - 1)

  ################ prediction step
  ######## prediction mean
  years_to_extrapolate <- as.numeric(parameter)
  if(years_to_extrapolate <=0)
    years_to_extrapolate <- len
  delta <- 1:len
  if(years_to_extrapolate < len)
    delta[years_to_extrapolate:len] <- years_to_extrapolate

  for(ct in Country){
    raw.data[paste(age1:age2),paste((year2+1):year3),ct] <- obj_all[[ct]]$a + as.matrix(obj_all[[ct]]$b) %*% delta
  }

  ######## prediction interval
  for(ct in Country){
    var.data[paste(age1:age2),paste((year2+1):year3),ct] <- as.matrix(obj_all[[ct]]$b)^2 %*%  (delta*sigma_Variance + c(1:len)*epsilon_Variance)
    raw.datalowerPI80[paste(age1:age2),paste((year2+1):year3),ct] <- raw.data[paste(age1:age2),paste((year2+1):year3),ct] - 1.281552*sqrt(var.data[paste(age1:age2),paste((year2+1):year3),ct])
    raw.datalowerPI95[paste(age1:age2),paste((year2+1):year3),ct] <- raw.data[paste(age1:age2),paste((year2+1):year3),ct] - 1.959964*sqrt(var.data[paste(age1:age2),paste((year2+1):year3),ct])
    raw.dataupperPI80[paste(age1:age2),paste((year2+1):year3),ct] <- raw.data[paste(age1:age2),paste((year2+1):year3),ct] + 1.281552*sqrt(var.data[paste(age1:age2),paste((year2+1):year3),ct])
    raw.dataupperPI95[paste(age1:age2),paste((year2+1):year3),ct] <- raw.data[paste(age1:age2),paste((year2+1):year3),ct] + 1.959964*sqrt(var.data[paste(age1:age2),paste((year2+1):year3),ct])
  }
  raw.data[raw.data<0] <- 0
  raw.datalowerPI80[raw.datalowerPI80<0] <- 0
  raw.datalowerPI95[raw.datalowerPI95<0] <- 0
  raw.dataupperPI80[raw.dataupperPI80<0] <- 0
  raw.dataupperPI95[raw.dataupperPI95<0] <- 0

  ######################################
  CPMobjlist <- list()
  for(ct in Country){
  ################ Output step
  predASFR <- raw.data[,,ct]
  predCASFR <- asfr_period_to_cohort(predASFR)

  predASFRlowerPI80 <- raw.datalowerPI80[,,ct]
  predASFRupperPI80 <- raw.dataupperPI80[,,ct]
  predCASFRlowerPI80 <- asfr_period_to_cohort(predASFRlowerPI80)
  predCASFRupperPI80 <- asfr_period_to_cohort(predASFRupperPI80)

  predASFRlowerPI95 <- raw.datalowerPI95[,,ct]
  predASFRupperPI95 <- raw.dataupperPI95[,,ct]
  predCASFRlowerPI95 <- asfr_period_to_cohort(predASFRlowerPI95)
  predCASFRupperPI95 <- asfr_period_to_cohort(predASFRupperPI95)

  # Use the country's full observed year range (including post-JOY years) so the
  # observed cohort completions match the Lee (1993) / Freeze Rates benchmarks.
  obsASFR=asfr_period_hfd[paste(age1:age2),,ct]
  obsCASFR=asfr_period_to_cohort(obsASFR)
  CPMobjlist[[ct]] <- list(pop=ct,
                            obsASFR=obsASFR,
                            obsCASFR=obsCASFR,
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
                            method="Myrskyla et al. (2013)",
                            parameter=parameter,
                            label=ifelse(all(!is.na(parameter)),paste("Myrskyla2013",paste(parameter,collapse = "_"),sep="_"),"Myrskyla2013"),
                            year=c(year1,year2,year3),
                            age=c(age1,age2),
                            obs=obs,
                            len=len,
                            cohort=c((year1-age2),(year1-age1),(year2-age2),(year2-age1)))
  }

  return(CPMobjlist)
}

###########################################
#### Prep ASFR data
# vector of required packages
required_packages <- c("tidyverse", "glue", "here")

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

# load and prepare data
path <- here("data")
asfr_training <- read.table(paste(path, "asfr_1891_to_2024.txt", sep = "/"),
                              header = FALSE)
# Drop the whole-country DEU / GBR series; their parts stay (see shared_helpers.R)
source(here("code/pnas_benchmarks/shared_helpers.R"))
asfr_training <- asfr_training[!(asfr_training[, 1] %in% excluded_country_idx(path)), ]
colnames(asfr_training) <- c('Country', 'Year', 'Age', 'Rate')

age1 <- 15
age2 <- 44
len  <- 30

joys <- c(1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013)

##############################
# BUILD 3D ARRAY (age x year x country)
##############################
# The Myrskyla method loops over countries internally to pool the prediction
# interval variance, so it needs a rectangular (age x year x country) array.
# Countries missing a given age/year are filled with NA; those NA propagate to
# the forecasts and are dropped when the long-format CSV is written.

ages   <- age1:age2
years  <- seq(min(asfr_training$Year), max(asfr_training$Year))
countries <- sort(unique(asfr_training$Country))

asfr_array <- array(NA,
                    dim = c(length(ages), length(years), length(countries)),
                    dimnames = list(as.character(ages), as.character(years), as.character(countries)))

# Keep only rows within the modeled age range, then place each rate at its
# (age, year, country) index in one vectorized assignment.
in_range <- asfr_training$Age %in% ages
idx <- cbind(match(asfr_training$Age[in_range], ages),
             match(asfr_training$Year[in_range], years),
             match(asfr_training$Country[in_range], countries))
asfr_array[idx] <- asfr_training$Rate[in_range]

min_data_year <- min(years)

##############################
# HELPER
##############################

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
cohort_pi_to_long <- function(item, country, joy, method, key) {
  mat <- item$predCASFR
  if (is.null(mat) || all(is.na(mat))) return(NULL)
  ages <- as.numeric(rownames(mat))
  cohorts <- as.numeric(colnames(mat))
  expand <- expand.grid(Age = ages, Year = cohorts)
  expand$Rate <- as.vector(as.matrix(mat))
  expand$Lower80 <- as.vector(as.matrix(item$predCASFRlowerPI80))
  expand$Upper80 <- as.vector(as.matrix(item$predCASFRupperPI80))
  expand$Lower95 <- as.vector(as.matrix(item$predCASFRlowerPI95))
  expand$Upper95 <- as.vector(as.matrix(item$predCASFRupperPI95))
  expand$Country <- country
  expand$JumpOffYear <- joy
  expand$Method <- method
  expand$Key <- key
  expand <- expand[!is.na(expand$Rate), c("Country", "Year", "Age", "Rate",
                                          "Lower80", "Upper80", "Lower95", "Upper95",
                                          "JumpOffYear", "Method", "Key")]
  return(expand)
}

all_forecasts <- list()
all_forecasts_pi <- list()
all_obs <- list()

##############################
# RUN EACH JUMP-OFF YEAR
##############################

for (joy in joys) {

  # obs spans the full observed history, matching Lee / Freeze Rates. The
  # Myrskyla model still fits its linear trend on the last 5 years internally.
  obs_years <- joy - min_data_year + 1

  tryCatch({
    result <- Method18_Myrskyla2013.R(
      asfr_period_hfd = asfr_array,
      joy   = joy,
      obs   = obs_years,
      age1  = age1,
      age2  = age2,
      parameter = c(5),
      len   = len
    )

    method_name <- "Myrskyla2013"

    for (ct in names(result)) {
      item <- result[[ct]]
      key_name <- paste0("Myrskyla2013_", ct)

      # Extract forecasted cohort ASFR
      fc <- cohort_matrix_to_long(item$predCASFR, ct, joy, method_name, key_name)
      if (!is.null(fc)) all_forecasts[[length(all_forecasts) + 1]] <- fc

      # Same forecast with its prediction interval bounds
      fc_pi <- cohort_pi_to_long(item, ct, joy, method_name, key_name)
      if (!is.null(fc_pi)) all_forecasts_pi[[length(all_forecasts_pi) + 1]] <- fc_pi

      # Extract observed cohort ASFR
      ob <- cohort_matrix_to_long(item$obsCASFR, ct, joy, method_name, key_name)
      if (!is.null(ob)) all_obs[[length(all_obs) + 1]] <- ob
    }

  }, error = function(e) {
    message(paste("Error in Jump-Off Year", joy, ":", e$message))
  })
}

# Combine and save as CSVs
forecasts_df <- bind_rows(all_forecasts)
obs_df <- collapse_obs(all_obs)   # one row per cell, no JumpOffYear (shared_helpers.R)

write.csv(forecasts_df, file = file.path(path, "myrskyla_forecasts_cohort.csv"), row.names = FALSE)
write.csv(obs_df, file = file.path(path, "myrskyla_obs_cohort.csv"), row.names = FALSE)

message(paste("Saved", nrow(forecasts_df), "forecast rows to", file.path(path, "myrskyla_forecasts_cohort.csv")))
message(paste("Saved", nrow(obs_df), "observed rows to", file.path(path, "myrskyla_obs_cohort.csv")))

# Forecasts with prediction intervals goes to its own file, leaving the two
# above untouched. 
forecasts_pi_df <- bind_rows(all_forecasts_pi)

write.csv(forecasts_pi_df, file = file.path(path, "myrskyla_forecasts_cohort_uncertainty.csv"), row.names = FALSE)

message(paste("Saved", nrow(forecasts_pi_df), "forecast rows with intervals to", file.path(path, "myrskyla_forecasts_cohort_uncertainty.csv")))
