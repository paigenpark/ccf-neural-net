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
#    Method: 
#    Method00_FreezeRates.R
############################################################
#### Input file: 
#     ASFR: age-specific fertility rate
#     
#     
#### Parameters
#      joy: jump of year
#      obs: length of observation
#      age1 : star age
#      age2 : end age
#      parameter : parameter of the method
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
#      
#### Ref:
#     
#
#   
###############################################################################
############################################
Method00_FreezeRates.R <- function(ASFR,
                                  joy = 1985, 
                                  obs = 30,
                                  age1 = 15, 
                                  age2 = 44,
                                  parameter = NA,
                                  len = 30,
                                  pop=""
){
  

    ################ Period to cohort function
  asfr_period_to_cohort <- function(input){
  temp <- expand.grid(as.numeric(dimnames(input)[[1]]),as.numeric(dimnames(input)[[2]]))
  cohort <- temp[,2] - temp[,1] # cohort = year - age
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
  # extract input data
  fert.raw <- ASFR[paste(age1:age2),paste(year1:year2)]
  raw.data <- matrix(NA,nrow=nrow(fert.raw),ncol=(year3-year1+1))
  colnames(raw.data) <- paste(year1 + c(1:ncol(raw.data))-1)
  rownames(raw.data) <- rownames(fert.raw)
  raw.data[,colnames(fert.raw)] <- fert.raw
  
  ########
  if(any(is.na(raw.data[,paste(year2)])))
  return(list(pop=pop,
              obsASFR=ASFR,
              obsCASFR=asfr_period_to_cohort(ASFR),
              predASFR=NA,
              predCASFR=NA,
              method="Freeze rate",
              parameter=parameter,
              label=ifelse(!is.na(parameter),paste("FreezeRate",parameter,sep="_"),"FreezeRate"),
              year=c(year1,year2,year3),
              age=c(age1,age2),
              obs=obs,
              len=len,
              cohort=c((year1-age2),(year1-age1),(year2-age2),(year2-age1))))
  
  ################ Model step
  #### freeze rate
  # takes in x, a vector representing fertility 
  # rates for one age group over time 
  projection.freeze <- function(x){
    ind <- !is.na(x)
    # which(ind) gets position numbers of all non missing values of x 
    # the largest position number (the last number in the sequence)
    # is used in every future year
    x[max(which(ind)+1):length(x)] <- x[max(which(ind))]
    return(x)
  }
  #### projection of parameter
  raw.data <- t(apply(raw.data,1,projection.freeze))
  
  ################ Output step
  predASFR=raw.data[,paste(year2+1:len)]
  predCASFR <- asfr_period_to_cohort(raw.data)
  #predCV <- predCASFR[,paste((year2-age1):(year2-age2+1))]
  
  ########
  return(list(pop=pop,
              obsASFR=ASFR,
              obsCASFR=asfr_period_to_cohort(ASFR),
              predASFR=predASFR,
              predCASFR=predCASFR,
              method="Freeze rates",
              parameter=parameter,
              label=ifelse(!is.na(parameter),paste("FreezeRates",parameter,sep="_"),"FreezeRates"),
              year=c(year1,year2,year3),
              age=c(age1,age2),
              obs=obs,
              len=len,
              cohort=c((year1-age2),(year1-age1),(year2-age2),(year2-age1))))
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
countries <- unique(asfr_training[,1])
ages <- unique(asfr_training[,3])
colnames(asfr_training) <- c('Country', 'Year', 'Age', 'Rate')

joys <- c(1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013)

all_forecasts <- list()
all_obs <- list()

##############################
# ROBUST LOOP WITH GAP CHECKS
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

  # 2. GAP CHECK: AGES
  required_ages <- as.character(15:44)
  missing_ages <- setdiff(required_ages, rownames(ASFR_mat))

  if (length(missing_ages) > 0) {
    message(paste("Skipping Country", i, "- Missing specific ages:", paste(head(missing_ages), collapse=",")))
    next
  }

  available_years <- as.numeric(colnames(ASFR_mat))
  min_data_year <- min(available_years)

  # 3. Iterate through Jump-Off Years
  for (joy in joys) {

    # Check bounds
    if (joy > max(available_years) || joy < min_data_year) {
      next
    }

    # 4. GAP CHECK: YEARS
    required_years_seq <- min_data_year:joy
    missing_years <- setdiff(as.character(required_years_seq), colnames(ASFR_mat))

    if (length(missing_years) > 0) {
      message(paste("Skipping", i, joy, "- Gap in data. Missing years:", paste(head(missing_years), collapse=",")))
      next
    }

    # Calculate obs and run
    obs_years <- joy - min_data_year + 1

    tryCatch({
      result <- Method00_FreezeRates.R(
        ASFR  = ASFR_mat,
        joy   = joy,
        obs   = obs_years,
        age1  = 15,
        age2  = 44,
        parameter = NA,
        len   = 30,
        pop   = i
      )

      method_name <- "FreezeRates"
      key_name <- paste0("FreezeRates_", i)

      # Extract forecasted cohort ASFR
      fc <- cohort_matrix_to_long(result$predCASFR, i, joy, method_name, key_name)
      if (!is.null(fc)) all_forecasts[[length(all_forecasts) + 1]] <- fc

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

write.csv(forecasts_df, file = file.path(path, "freeze_forecasts_cohort.csv"), row.names = FALSE)
write.csv(obs_df, file = file.path(path, "freeze_obs_cohort.csv"), row.names = FALSE)

message(paste("Saved", nrow(forecasts_df), "forecast rows to", file.path(path, "freeze_forecasts_cohort.csv")))
message(paste("Saved", nrow(obs_df), "observed rows to", file.path(path, "freeze_obs_cohort.csv")))
