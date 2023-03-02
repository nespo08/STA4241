## Calculation to find limiting values

## Generate a massive data set that approximates asymptotics
n = 1000000

## Draw skewed error distribution
error = rgamma(n, 1, 100)
error = (error - mean(error)) / sd(error)

hist(error)

## Generate large data set
x = rnorm(n)
y = x*2 + x^3 + error

data = data.frame(x=x, y=y)

## See what the coefficients are
linMod = lm(y ~ x, data=data)
linMod$coefficients
summary(linMod)

## Simulation to assess the performance of the analytic standard errors
n = 100

nSim = 1000

## Keep track of estimates, standard errors, and coverages
est = se = cov = rep(NA, nSim)

for (ni in 1 : nSim) {
  ## Generate data
  error = rgamma(n, 1, 100)
  error = (error - mean(error)) / sd(error)
  
  x = rnorm(n)
  y = x*2 + x^3 + error
  
  data = data.frame(x=x, y=y)
  
  ## Fit model to it
  linMod = lm(y ~ x, data=data)
  
  est[ni] = linMod$coefficients[2]
  se[ni] = summary(linMod)$coefficients[2,2]
  
  ## Confidence intervals from analytic expressions
  cov[ni] = 1*(est[ni] - 1.96*se[ni] < 5 &
                 est[ni] + 1.96*se[ni] > 5) 
}

## Plot the sampling distribution
hist(est)

## Mean of the sampling distribution
mean(est, na.rm=TRUE)

## Standard deviation of the sampling distribution
sd(est, na.rm=TRUE)

## Average value of our analytic SE estimates
mean(se, na.rm=TRUE)

## Empirical coverage probability
mean(cov, na.rm=TRUE)


## Now try three types of bootstrap to estimate standard errors
## Parametric, residual, nonparametric
nSim = 200

## Only 100 bootstrap samples for now to save time
## Generally do more like 1000
nBoot = 100

est = rep(NA, nSim)

## 4 columns, one for each bootstrap + the analytic standard errors
se = cov = matrix(NA, nSim, 4)

for (ni in 1 : nSim) {
  
  if (ni %% 10 == 0) print(ni)
  
  ## Generate data from nonlinear model
  error = rgamma(n, 1, 100)
  error = (error - mean(error)) / sd(error)
  
  x = rnorm(n)
  y = x*2 + x^3 + error
  
  data = data.frame(x=x, y=y)
  
  ## Estimate model
  linMod = lm(y ~ x, data=data)
  
  est[ni] = linMod$coefficients[2]
  
  ## Analytic standard error
  se[ni,1] = summary(linMod)$coefficients[2,2]

  ## Coverage for analytic standard error
  cov[ni,1] = 1*(est[ni] - 1.96*se[ni] < 5 &
                 est[ni] + 1.96*se[ni] > 5) 
  
  ## Now bootstrap to get standard errors
  ## Keep 3 columns, one for each bootstrap type
  estBoot = matrix(NA, nBoot, 3)
  for (nb in 1 : nBoot) {
    ## Parametric bootstrap first
    # get estimates of all unknown parameters from original data
    coefs = linMod$coefficients
    residSE = sqrt(sum(linMod$residuals^2) / (n - 2))
    
    # Now draw a new outcome vector from this model
    yNew = coefs[1] + coefs[2]*data$x + rnorm(n, sd=residSE)
    
    # create new data set for parametric bootstrap
    dataBoot1 = data.frame(x = data$x,
                           y = yNew)
    
    # fit model to this data
    linModBoot1 = lm(y ~ x, data=dataBoot1)
    estBoot[nb,1] = linModBoot1$coefficients[2]
    
    ## Residual bootstrap
    
    # randomly sample from the residual distribution
    samp = sample(1:n, n, replace=TRUE)
    residuals = linMod$residuals[samp]
    
    yNew2 = coefs[1] + coefs[2]*data$x + residuals
    
    # create new data set for parametric bootstrap
    dataBoot2 = data.frame(x = data$x,
                           y = yNew2)
    
    # fit model to this data
    linModBoot2 = lm(y ~ x, data=dataBoot2)
    estBoot[nb,2] = linModBoot2$coefficients[2]

    ## Nonparametric bootstrap
    samp = sample(1:n, n, replace=TRUE)
    dataBoot3 = data[samp,]
    
    linModBoot3 = lm(y ~ x, data=dataBoot3)
    estBoot[nb,3] = linModBoot3$coefficients[2]
  }
  
  ## Take the standard deviation of bootstrap samples as
  ## Estimates of the standard error
  bootstrapSEs = apply(estBoot, 2, sd)
  
  se[ni,2:4] = bootstrapSEs
  
  cov[ni,2:4] = 1*(est[ni] - 1.96*se[ni,2:4] < 5 &
                     est[ni] + 1.96*se[ni,2:4] > 5)
}

## Mean of sampling distribution
mean(est, na.rm=TRUE)

## Standard deviation of sampling distribution
sd(est, na.rm=TRUE)

## Average estimated standard error for each approach
apply(se, 2, mean, na.rm=TRUE)

## Empirical coverage probability
apply(cov, 2, mean, na.rm=TRUE)


## Does the nonparametric bootstrap add substantial uncertainty by
## resampling the X values as well?
nSim = 2000
nBoot = 100

est = rep(NA, nSim)

## Compare the analytic standard errors with the bootstrap ones
se = cov = matrix(NA, nSim, 2)

for (ni in 1 : nSim) {
  
  if (ni %% 10 == 0) print(ni)
  
  x = rnorm(n)
  
  ## Generate data from linear model
  error = rnorm(n)
  
  y = 2*x + error
  
  data = data.frame(x=x, y=y)
  
  ## Estimate model
  linMod = lm(y ~ x, data=data)
  
  est[ni] = linMod$coefficients[2]
  
  ## Analytic standard error
  se[ni,1] = summary(linMod)$coefficients[2,2]
  
  ## Coverage for analytic standard error
  cov[ni,1] = 1*(est[ni] - 1.96*se[ni,1] < 2 &
                   est[ni] + 1.96*se[ni,1] > 2) 
  
  ## Now bootstrap to get standard errors
  estBoot = rep(NA, nBoot)
  for (nb in 1 : nBoot) {
    ## Nonparametric bootstrap
    samp = sample(1:n, n, replace=TRUE)
    dataBoot = data[samp,]
    
    linModBoot = lm(y ~ x, data=dataBoot)
    estBoot[nb] = linModBoot$coefficients[2]
  }
  
  ## Take the standard deviation of bootstrap samples as
  ## Estimates of the standard error
  bootstrapSE = sd(estBoot)
  
  se[ni,2] = bootstrapSE
  
  cov[ni,2] = 1*(est[ni] - 1.96*se[ni,2] < 2 &
                     est[ni] + 1.96*se[ni,2] > 2)
}

## Mean of sampling distribution
mean(est, na.rm=TRUE)

## Standard deviation of sampling distribution
sd(est, na.rm=TRUE)

## Average estimated standard error for each approach
apply(se, 2, mean, na.rm=TRUE)

## Empirical coverage probability
apply(cov, 2, mean, na.rm=TRUE)


