library(mvtnorm)

## Very large sample size to take away sampling variability
n = 100000

## Vary the correlation among the two covariates
rhoVec = seq(0, 0.99, by=0.01)

## Save estimated coefficients from models
beta2Save = matrix(NA, length(rhoVec), 2)

## Loop through different correlation values
for (jj in 1 : length(rhoVec)) {
  rho = rhoVec[jj]
  
  ## Covariance matrix of the two covariates
  sigma = matrix(rho, 2, 2)
  diag(sigma) = 1
  
  ## Generate two covariates
  X = rmvnorm(n, sigma=sigma)
  
  ## Generate outcome
  meanY = 5 + 2*X[,1] + 0*X[,2]
  Y = rnorm(n, meanY, sd=1)
  
  ## Fit joint model and model that only includes X2
  jointModel = lm(Y ~ X)
  simpleModel = lm(Y ~ X[,2])
  
  ## Store the results
  beta2Save[jj,1] = jointModel$coefficients[3]
  beta2Save[jj,2] = simpleModel$coefficients[2]
}

## Plot coefficients as a function of the correlation
plot(rhoVec, beta2Save[,1], type="l", lwd=3, 
     xlab=expression(rho), ylab = expression(beta[2]),
     ylim=range(beta2Save))
lines(rhoVec, beta2Save[,2], lwd=3, col=2)
legend("topleft", c("Joint model", "Simple model"), lty=1, lwd=3, col=1:2)



## Simulation that shows the type I error rate
n = 200

## Run 1000 simulated data sets
nSim = 1000

## Store whether the null hypothesis is rejected
rejects = matrix(NA, nSim, 2)

for (i in 1 : nSim) {
  ## Generate data with 40 covariates
  x = rmvnorm(n, sigma=diag(40))
  
  ## Generate outcome, independently of X
  y = rnorm(n)
  
  ## first perform F-test
  mod = lm(y ~ x)
  anov = anova(mod)
  
  ## Store p-value
  pval1 = anov$`Pr(>F)`[1]
  rejects[i,1] = 1*(pval1 < 0.05)
  
  ## now use individual tests. Check if any p-values are below 0.05
  rejects[i,2] = 1*(any(summary(mod)$coefficients[,4] < 0.05))
}

## Calculate the type I error rate
apply(rejects, 2, mean)




## Let's look at R2 and RSE as a function 
## of the number of covariates included 

x = rmvnorm(n, sigma=diag(40))
y = 2*x[,1] + 1*x[,2] + rnorm(n)

## Store the R2 and RSE values
r2 = rep(NA, 41)
rse = rep(NA, 41)

## First look at model that does not include any covariates
mod = lm(y ~ 1)
r2[1] = cor(y, mod$fitted.values)^2
rse[1] = sum((y - mod$fitted.values)^2) / (n-1)

## Loop through including increasing numbers of covariates
for (i in 1 : 40) {
  mod = lm(y ~ x[,1:i])
  r2[i+1] = cor(y, mod$fitted.values)^2
  rse[i+1] = sum((y - mod$fitted.values)^2) / (n-i-1)
}

plot(0:40, r2, type="l", lwd=3, xlab="Variables included", ylab="Variance explained",
     main="Variance explained")
plot(0:40, rse, type="l", lwd=3, xlab="Variables included", ylab="RSE",
     main="RSE")


## Now let's look at polynomial regression with one covariate
n = 1500
x = rnorm(n, mean=2)

## Generate outcome from highly nonlinear model
y = 0.8*exp(0.65*x) + rnorm(n)

## plot function at a sequence of points
s = seq(min(x), max(x), length=1000)

## plot data and true curve
plot(x, y, type='n', main="Truth is nonlinear")
lines(s, 0.8*exp(0.65*s), type='l', lwd=3)

## Now fit linear model
linMod = lm(y ~ x)
predictions1 = predict(linMod, newdata = data.frame(x=s))

## q = 3 degree of freedom polynomials
linMod3 = lm(y ~ poly(x, 3))
predictions3 = predict(linMod3, newdata = data.frame(x=s))

## q = 5 degree of freedom polynomials
linMod5 = lm(y ~ poly(x, 5))
predictions5 = predict(linMod5, newdata = data.frame(x=s))

## Plot predicted fits for all three models
lines(s, predictions1, lwd=3, col=2)
lines(s, predictions3, lwd=3, col=3)
lines(s, predictions5, lwd=3, col=4)
legend("topleft", c("Truth", "q = 1", "q = 3", "q = 5"), lty=1, lwd=3, col=1:4)


## Do the same thing, but on a data set where truth is linear
n = 1500
x = rnorm(n, mean=2)
y = 0.5*x + rnorm(n)

s = seq(min(x), max(x), length=1000)

plot(x, y, type='n', main="Truth is linear")
lines(s, 0.5*s, type='l', lwd=3)

## Now fit linear model
linMod = lm(y ~ x)
predictions1 = predict(linMod, newdata = data.frame(x=s))

## q = 3 degree of freedom polynomials
linMod3 = lm(y ~ poly(x, 3))
predictions3 = predict(linMod3, newdata = data.frame(x=s))

## q = 5 degree of freedom polynomials
linMod5 = lm(y ~ poly(x, 5))
predictions5 = predict(linMod5, newdata = data.frame(x=s))

lines(s, predictions1, lwd=3, col=2)
lines(s, predictions3, lwd=3, col=3)
lines(s, predictions5, lwd=3, col=4)
legend("topleft", c("Truth", "q = 1", "q = 3", "q = 5"), lty=1, lwd=3, col=1:4)
