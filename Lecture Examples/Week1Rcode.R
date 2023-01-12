rm(list=ls())

## Code that runs a simulation to visualize the sampling distribution

## Number of simulations
nSim = 1000

## Store the results
est = rep(NA, nSim)

## Each time generate new data and calculate the sample mean
for (i in 1 : nSim) {
  x = rnorm(10)
  est[i] = mean(x)
}

## plot a histogram of the results
hist(est, col="darkgrey", main="Sampling distribution", 
     xlab=expression(theta), probability=TRUE)

## Find the mean and variance of the estimates
mean(est)
var(est)


## Simulation that finds the reducible error
nSim = 1000

## Vary sample size from 5 to 100
nVec = seq(5,100, by=5)

## store results
error1 = matrix(NA, nSim, length(nVec))

for (i in 1 : nSim) {
  for (ni in 1 : length(nVec)) {
    n = nVec[ni]
    
    ## generate data from a simple linear model
    x = rnorm(n)
    y = x*2 + rnorm(n)
    
    ## Fit the linear model and store the coefficient
    mod = lm(y ~ -1 + x)
    error1[i,ni] =  (mod$coefficients[1] - 2)^2
  }
}

## Find the MSE for each sample size
error1overall = apply(error1, 2, mean, na.rm=TRUE)

## Plot the results as a function of sample size
plot(nVec, error1overall, type='l', lwd=4, main="Error types",
     xlab="Sample size", ylab = "Squared error", ylim=c(0,1.2))

## Plot a horizontal line for the irreducible error
abline(h = 1, col=2, lwd=4)

## Make a legend
legend("right", c("Reducible", "Irreducible"), col=1:2, lwd=4, cex=1.6)


## Illustration of overfitting on a linear model

## load the splines library
library(splines)

## sample size
n = 100

## generate data from a simple linear model
x = rnorm(n)
y = x + rnorm(n)

## store data in a data frame
dat = data.frame(x=x, y=y)

## generate new data points to predict at
xnew = data.frame(x=seq(-2,2, length=500))
ynew = as.numeric(xnew$x) + rnorm(500)

## fit linear model and extract predictions
linMod = lm(y ~ x, data=dat)
predLin = predict(linMod, newdata=xnew)

## fit 5 degree of freedom splines
splineMod5 = lm(y ~ ns(x, 5), data=dat)
predSplines5 = predict(splineMod5, newdata=xnew)

## fit 50 degree of freedom splines
splineMod50 = lm(y ~ ns(x, 50), data=dat)
predSplines50 = predict(splineMod50, newdata=xnew)

## In sample MSE
MSE1 = mean((y - predict(linMod))^2)
MSE2 = mean((y - predict(splineMod5))^2)
MSE3 = mean((y - predict(splineMod50))^2)

## Out of sample MSE
MSE1new = mean((ynew - predict(linMod, newdata=xnew))^2)
MSE2new = mean((ynew - predict(splineMod5, newdata=xnew))^2)
MSE3new = mean((ynew - predict(splineMod50, newdata=xnew))^2)

## Compare MSE for in and out of sample
MSE1
MSE2
MSE3

MSE1new
MSE2new
MSE3new

## Plot the predicted fits
plot(xnew$x, predLin, type='l', lwd=3, col=2, ylim=range(dat$y),
     xlab="Predictor", ylab="Estimate of f", main="Parametric to nonparametric fits")
abline(0,1, lty=1, lwd=3)
lines(xnew$x, predSplines5, col=3, lwd=3)
lines(xnew$x, predSplines50, col=4, lwd=3)
legend("topleft", c("Truth","Linear", "Somewhat flexible", "Highly flexible"), col=1:4, lwd=3)
points(dat$x, dat$y, pch=16, cex=0.5)
dev.off()


## Let's do the same thing but with a nonlinear f
n = 100
x = rnorm(n)

## Nonlinear function f
ftrue = function(x) {
  return(0.5*x + 0.2*x^4 + 0.2*exp(x) - 4*sin(x*pi))
}
y = ftrue(x) + rnorm(n)

## storing data in a data frame
dat = data.frame(x=x, y=y)

## new x values to predict at
xnew = data.frame(x=seq(-2,2, length=500))

## Fit the 3 different models again
linMod = lm(y ~ x, data=dat)
predLin = predict(linMod, newdata=xnew)

splineMod5 = lm(y ~ ns(x, 5), data=dat)
predSplines5 = predict(splineMod5, newdata=xnew)

splineMod50 = lm(y ~ ns(x, 50), data=dat)
predSplines50 = predict(splineMod50, newdata=xnew)

## Plot predicted fits
plot(xnew$x, predLin, type='l', lwd=3, col=2, ylim=range(dat$y),
     xlab="Predictor", ylab="Estimate of f", main="Parametric to nonparametric fits")
lines(xnew$x, ftrue(xnew$x), col=1, lwd=3)
lines(xnew$x, predSplines5, col=3, lwd=3)
lines(xnew$x, predSplines50, col=4, lwd=3)
legend("topleft", c("Truth","Linear", "Somewhat flexible", "Highly flexible"), col=1:4, lwd=3)
points(dat$x, dat$y, pch=16, cex=0.5)

