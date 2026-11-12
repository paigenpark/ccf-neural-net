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
#    2016.10.04
#    
#    Method: deBeer 1985 and 1989 
#    Method13_deBeer1985and1989.R
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
#      parameter : orderage and ordercht
#             1. orderage: parameter[1:3] order of age
#             2. ordercht: parameter[4:6] order of cohort
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
#
#### NOTE:
#      
#### Ref:
#  1. de Beer J (1985) A Time Series Model for Cohort Data. Journal of the American Statistical Association 80(391):525-530.     
#  2. de Beer J (1989) Projecting age-speci???c fertility rates by using time-series methods. European Journal of Population 5:315-346.  
#  
###############################################################################
# vector of required packages
required_packages <- c("minpack.lm", "tidyverse", "glue", "here")

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

################################
LAG <- function(xy,lagx=1,lagy=1){
  if(!is.list(xy)){
    xseq <- c(1:nrow(xy)) - lagx
    yseq <- c(1:ncol(xy)) - lagy
    out <- list(mts = xy, xseq=xseq, yseq=yseq)
  }
  if(is.list(xy)){
    xy$xseq <- xy$xseq - lagx
    xy$yseq <- xy$yseq - lagy
    out <- xy
  }
  return(out)
}

mts.intersect <- function(xy1,xy2){
  xseq <- intersect(xy1$xseq,xy2$xseq)
  yseq <- intersect(xy1$yseq,xy2$yseq)
  name.xy1 <- dimnames(xy1$mts)
  name.xy2 <- dimnames(xy2$mts)
  xy1$mts <- matrix(xy1$mts[xy1$xseq %in% xseq, xy1$yseq %in% yseq],nrow=length(xseq),ncol=length(yseq))
  dimnames(xy1$mts) <- list(name.xy1[[1]][xy1$xseq %in% xseq],name.xy1[[2]][xy1$yseq %in% yseq])
  xy1$xseq = xseq
  xy1$yseq = yseq
  xy2$mts <- matrix(xy2$mts[xy2$xseq %in% xseq, xy2$yseq %in% yseq],nrow=length(xseq),ncol=length(yseq))
  dimnames(xy2$mts) <- list(name.xy2[[1]][xy2$xseq %in% xseq],name.xy2[[2]][xy2$yseq %in% yseq])
  xy2$xseq = xseq
  xy2$yseq = yseq
  
  return(list(xy1=xy1,xy2=xy2))
}


mts.merge <- function(xy1,xy2){
  xseq <- intersect(xy1$xseq,xy2$xseq)
  yseq <- intersect(xy1$yseq,xy2$yseq)
  xy1$mts[xy1$xseq %in% xseq, xy1$yseq %in% yseq] <- xy2$mts[xy2$xseq %in% xseq, xy2$yseq %in% yseq]
  return(xy1)
}

inverse.diff <- function(x){
  if(length(x)==1)
    return(x)
  for(i in 2:length(x)){
    x[i] <- x[i] + x[i-1]
  }
  return(x)
}

######################################################
#### function check autocorrelation coeficients

acf.matrix <- function(da,lagx.max=NULL,lagy.max=NULL,type=c("correlation")){
  if(is.null(lagx.max)){
    lagx.max <- nrow(da)-1
  }
  if(is.null(lagy.max)){
    lagy.max <- ncol(da)-1
  }
  ####
  if(lagx.max > nrow(da)-1)
    lagx.max <- nrow(da)-1
  if(lagy.max > ncol(da)-1)
    lagy.max <- ncol(da)-1
  ####
  da <- da - mean(da)
  da.s <- sum(da^2)
  ####
  out <- matrix(NA,nrow=lagx.max+1,ncol=lagy.max+1)
  if("correlation" %in% type){
    for(i in 0:lagx.max+1){
      for(j in 0:lagy.max+1){
        temp1 <- da[c(1:c(nrow(da)-i+1)),c(1:c(ncol(da)-j+1))]
        temp2 <- da[c(i:nrow(da)),c(j:ncol(da))]
        out[i,j] <- sum(temp1*temp2) / da.s
      }
    }
  }
  
  return(out)
}

#############################################################
#### ARIMA model using Levenberg-Marquardt
#### nonlinear least-squares algorithm 
#### For CARIMA model proposed by de Beer 1985
#### This version is used for CARIMA model with q1=q2=0
#### MA part is under developing
############################
ts.matrix.nls.lm.full <- function(x,orderx=c(1,0,0),ordery=c(1,0,0),h=1){
  #### initial data
  xx <- as.matrix(x)
  if(orderx[2]>0){
    xx <- diff(x,orderx[2])
  }
  if(ordery[2]>0){
    xx <- t(diff(t(x),ordery[2]))
  }
  xx <- LAG(xx,0,0)
  xl <- xx
  
  ## parameters
  parStart <- c(rep(0.1,orderx[1]), rep(0.1,ordery[1]))
  
  ## model 
  model.arima <- function(parS,xl){
    
    partempx <- c(1)
    partempy <- c(1)
    
    if(orderx[1]>0)
      partempx <- c(1,parS[1:orderx[1]])
    if(ordery[1]>0)
      partempy <- c(1,parS[orderx[1]+c(1:ordery[1])])
    
    out <- xl
    out$mts <- -out$mts
    for(i in 0:orderx[1]){
      for(j in 0:ordery[1]){
        temp = LAG(xl,-i,-j)
        opt <- mts.intersect(out,temp)
        out <- opt$xy1
        temp <- opt$xy2
        out$mts <- out$mts + partempx[i+1] * partempy[j+1] * temp$mts
      }
    }
    return(out)
  }
  ## residual function
  residFun <- function(p, observed, xl){ 
    opt <- mts.intersect(observed,model.arima(p,xl))
    res <- as.vector(opt$xy1$mts - opt$xy2$mts)
    return(res[!is.na(res)]) 
  }
  
  nls.out <- nls.lm(par=parStart, fn = residFun, observed = xl,
                    xl = xl, control = nls.lm.control(nprint=F))
  #### compute forecasting part
  
  ## add one column
  add.one.col <- function(xl,num){
    xl$mts <- cbind(xl$mts,num)
    colnames(xl$mts)[ncol(xl$mts)] <- as.character(as.numeric(colnames(xl$mts)[ncol(xl$mts)-1]) +1)
    xl$yseq <- c(xl$yseq,xl$yseq[length(xl$yseq)] + 1)
    xl
  }
  ## forecast 
  forecast.arima <- function(parS,xl){
    partempx <- c(1,parS[1:orderx[1]])
    partempy <- c(1,parS[orderx[1]+c(1:ordery[1])])
    
    last <- max(which(!apply(xl$mts,2,function(x){all(is.na(x))})))

    if(ncol(xl$mts)<= last){
      out <- add.one.col(xl,0)
      xxl <- add.one.col(xl,0)
    }
    if(ncol(xl$mts) > last){
      out <- xl
      xxl <- xl
    }
    
    OUT <- out
    for(l in (last+1):ncol(out$mts)){
      for(i in 0:orderx[1]){
        for(j in 0:ordery[1]){
          temp = LAG(xxl,-i,-j)
          opt <- mts.intersect(out,temp)
          out <- opt$xy1
          temp <- opt$xy2
          
          out$mts[,l] <- out$mts[,l] + partempx[i+1] * partempy[j+1] * temp$mts[,l-1]
          OUT <- mts.merge(OUT,out)
        }
      }
    }

    return(OUT)
  }
  ## forecast for h 
  parEst <- nls.out$par
  #for(i in 1:h){
  #  xl <- forecast.arima(parEst,xl)
  #}
  
  #### forecast and inverse differences for cohort
  if(orderx[1]>1){
    for(i in 2:orderx[1]){
      for(j in ncol(xx$mts)-c((i-2):0))
        xx$mts[i,j] <- xx$mts[i,ncol(xx$mts)-i+1]
    }
  }
  ##
  last <- max(which(!apply(xx$mts,2,function(x){all(is.na(x))})))
  if(ordery[1]>0){
    partempx <- c(parEst[orderx[1]:1],1)
    partempy <- c(parEst[orderx[1]+c(ordery[1]:1)],1)
    
    for(j in (last+1):ncol(xx$mts)){
      temp <- c(xx$mts[1,(j-ordery[1]):j])
      temp[is.na(temp)] <- 0
      xx$mts[1,j] <- temp %*% partempy
    }
    
    for(i in max(2,orderx[1]):nrow(xx$mts)){
      for(j in ((2-i)+last):ncol(xx$mts)){
        temp <- matrix(xx$mts[(i-orderx[1]):i,(j-ordery[1]):j],nrow=orderx[1]+1,ncol=ordery[1]+1)
        temp[is.na(temp)] <- 0
        xx$mts[i,j] <- partempx %*% temp %*% partempy
      }
    }
  }
  ##
  ind <- is.na(x)
  for(i in 1:nrow(xx$mts)-1){
    for(j in 1:ncol(xx$mts)-1){
      if(is.na(x[nrow(x)-i,ncol(x)-j])){
        x[nrow(x)-i,ncol(x)-j] <- xx$mts[nrow(xx$mts)-i,ncol(xx$mts)-j]
      }
    }
  }
  
  if(ordery[2]>0){
    for(i in 1:nrow(x)){
      x[i,max(which(ind[i,] == F)):ncol(x)] <- inverse.diff(x[i,max(which(ind[i,] == F)):ncol(x)])
    }
  }
  
  if(orderx[2]>0){
    for(j in 1:ncol(x)){
      x[max(which(ind[,j] == F)):nrow(x),j] <- inverse.diff(x[max(which(ind[,j] == F)):nrow(x),j])
    }
  }
  #### return
  return(list(obj=nls.out,
              pred = x,
              #rs=matrix(residuals(nls.out),nrow=nrow(x)-orderx[1] - orderx[2], ncol=ncol(x) - ordery[1] - ordery[2]))
              rs=residuals(nls.out)
  ))
}

##################################################################
############################################
Method13_deBeer1985and1989.R <- function(ASFR,
                                         joy = 1985, 
                                         obs = 30,
                                         age1 = 15, 
                                         age2 = 44,
                                         parameter = c(1,0,0,2,1,0),
                                         len = 30,
                                         pop=""
){
  ################ Period to cohort function
  asfr_period_to_cohort <- function(input){
  input <- as.matrix(input)
  temp <- expand.grid(as.numeric(dimnames(input)[[1]]),as.numeric(dimnames(input)[[2]]))
  cohort <- temp[,2] - temp[,1]
  output <- tapply(as.vector(input), list(temp[,1], cohort),sum)
  return(as.data.frame(output))
}
  ################ Cohort to period function
  asfr_cohort_to_period <- function(input){
  input <- as.matrix(input)
  temp <- expand.grid(as.numeric(dimnames(input)[[1]]),as.numeric(dimnames(input)[[2]]))
  period <- temp[,1] + temp[,2]
  output <- tapply(as.vector(input), list(temp[,1], period),sum)
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
  fert.raw[fert.raw<=0.00001] <- 0.00001
  raw.data <- matrix(NA,nrow=nrow(fert.raw),ncol=(year3-year1+1))
  colnames(raw.data) <- paste(year1 + c(1:ncol(raw.data))-1)
  rownames(raw.data) <- rownames(fert.raw)
  raw.data[,colnames(fert.raw)] <- fert.raw
  
  # age*period to age*cohort 
  raw.cdata <- asfr_period_to_cohort(raw.data)
  

  ########
  if(any(is.na(sum(raw.data[,paste(c(year1,year2))]))) | !any(apply(raw.cdata,2,function(x) all(!is.na(x)))) )
  return(list(pop=pop,
              obsASFR=ASFR,
              obsCASFR=asfr_period_to_cohort(ASFR),
              predASFR=NA,
              predCASFR=NA,
              method="de Beer (1985 and 1989)",
              parameter=parameter,
              label=ifelse(all(!is.na(parameter)),paste("deBeer",paste(parameter,collapse = "_"),sep="_"),"deBeer"),
              year=c(year1,year2,year3),
              age=c(age1,age2),
              obs=obs,
              len=len,
              cohort=c((year1-age2),(year1-age1),(year2-age2),(year2-age1))))
  
  ################ Model step
  nls.out <- ts.matrix.nls.lm.full(raw.cdata,orderx=parameter[1:3],ordery=parameter[4:6])
  
  ################ Output step
  predCASFR <- nls.out$pred
  predASFR <- asfr_cohort_to_period(predCASFR)

  ########
  return(list(pop=pop,
              obsASFR=ASFR,
              obsCASFR=asfr_period_to_cohort(ASFR),
              predASFR=predASFR,
              predCASFR=predCASFR,
              method="de Beer (1985 and 1989)",
              parameter=parameter,
              label=ifelse(all(!is.na(parameter)),paste("deBeer",paste(parameter,collapse = "_"),sep="_"),"deBeer"),
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
  # The function strictly requires ages 15 through 44.
  required_ages <- as.character(15:44)
  missing_ages <- setdiff(required_ages, rownames(ASFR_mat))

  if (length(missing_ages) > 0) {
    message(paste("Skipping Country", i, "- Missing specific ages:", paste(head(missing_ages), collapse=",")))
    next
  }

  available_years <- as.numeric(colnames(ASFR_mat))
  min_data_year <- min(available_years)

  # de Beer 2 uses the best-performing observation window from the paper: 40 years.
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
      result <- Method13_deBeer1985and1989.R(
        ASFR  = ASFR_mat,
        joy   = joy,
        obs   = obs_years,
        age1  = 15,
        age2  = 44,
        parameter = c(1, 0, 0, 2, 1, 0),
        len   = 30,
        pop   = i
      )

      # STABILITY CHECK: nls.lm fits the CARIMA coefficients with no stationarity
      # constraint, so some fits give an explosive recursion (e.g. Japan, JOYs
      # 1986-1990: 40-year window starting in the 1947-51 post-war boom, plus the
      # 1966 fire-horse dip, gives forecasts of +/-1e6). Treat any forecast outside
      # a loose plausible range as a failed fit and skip it, like the gap checks.
      # Skipped cells also drop out of the eval's common grid for every model,
      # matching the paper's equal-testing-dataset approach.
      # Only cells inside the forecast horizon (period <= JOY + len) are checked:
      # the cohort matrix also holds cells beyond it (young cohorts at old ages),
      # which are never scored and can be noisy even in an otherwise sound fit.
      pred_mat <- as.matrix(result$predCASFR)
      period <- outer(as.numeric(rownames(pred_mat)), as.numeric(colnames(pred_mat)), "+")
      pred_vals <- pred_mat[!is.na(pred_mat) & period <= joy + 30]
      stable <- length(pred_vals) > 0 && all(is.finite(pred_vals)) &&
        all(pred_vals >= -0.5 & pred_vals <= 1.5)

      method_name <- "deBeer"
      key_name <- paste0("deBeer_", i)

      # Extract forecasted cohort ASFR
      if (stable) {
        fc <- cohort_matrix_to_long(result$predCASFR, i, joy, method_name, key_name)
        if (!is.null(fc)) all_forecasts[[length(all_forecasts) + 1]] <- fc
      } else {
        message(sprintf("Skipping Country %s JOY %d - unstable fit (forecast range %.3g to %.3g)",
                        i, joy, min(pred_vals), max(pred_vals)))
      }

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

write.csv(forecasts_df, file = file.path(path, "debeer2_forecasts_cohort.csv"), row.names = FALSE)
write.csv(obs_df, file = file.path(path, "debeer2_obs_cohort.csv"), row.names = FALSE)

message(paste("Saved", nrow(forecasts_df), "forecast rows to", file.path(path, "debeer2_forecasts_cohort.csv")))
message(paste("Saved", nrow(obs_df), "observed rows to", file.path(path, "debeer2_obs_cohort.csv")))

