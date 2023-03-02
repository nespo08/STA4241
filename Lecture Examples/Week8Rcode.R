library(glmnet)
library(spls)
library(leaps)
library(pls)
library(mvtnorm)

## Load the mice data
data(mice)

## Create a data frame
data = data.frame(y = mice$y[,1], x=mice$x[,11:20])

## Find ridge regression estimates (alpha = 0)
fitRidge = glmnet(x = mice$x[,11:20], y = mice$y[,1], alpha = 0) # X, Y must be vector
plot(fitRidge, xvar="lambda", xlab="Log Lambda", main="Ridge estimates")

## CV curve for ridge regression
fitRidgeCV = cv.glmnet(x = mice$x[,11:20], y = mice$y[,1], alpha = 0)
plot(fitRidgeCV)

## Lasso estimates
fitLasso = glmnet(x = mice$x[,11:20], y = mice$y[,1], alpha = 1)
plot(fitLasso, xvar="lambda", xlab="Log Lambda", main="Lasso estimates")

## CV curve for lasso
fitLassoCV = cv.glmnet(x = mice$x[,11:20], y = mice$y[,1], alpha = 1)
plot(fitLassoCV)

## PCR example
n = 60
p = 50

## Correlation matrix
rho = 0.9
sigma = matrix(NA, p, p)
for (i in 1 : nrow(sigma)) {
  for (j in 1 : nrow(sigma)) {
    sigma[i,j] = (rho^abs(i-j))
  }
}

## Show covariance matrix
image(sigma)

## Run simulation study
M = 1000

## Testing data set sample size
n_test = 100

## Store errors for OLS and PCR approaches
error = matrix(NA, M, 2)
for (m in 1 : M) {
  ## Generate covariates
  x = mvtnorm::rmvnorm(n, sigma=sigma)
  
  ## Generate random coefficients and outcome
  beta = rnorm(p, sd=0.5)
  y = x %*% beta + rnorm(n)
  
  data = data.frame(y=y, x=x)
  
  ## Generate new data
  xnew = mvtnorm::rmvnorm(n_test, sigma=sigma)
  ynew = xnew %*% beta + rnorm(n_test)
  
  data_test = data.frame(y=ynew, x=xnew)
  
  
  ## Run PCR where we keep the first 10 components
  pcr = pcr(y ~ ., data=data, scale=TRUE, ncomp=10)
  
  pcr.pred = predict(pcr, data_test, ncomp = 10)
  error[m,1] = mean((as.numeric(pcr.pred) - data_test$y)^2)
  
  ## Regular least squares
  mod2 = lm(y ~ ., data=data)
  
  ## Make predictions
  pred2 = predict(mod2, newdata = data_test)
  
  ## Store MSE
  error[m,2] = mean((pred2 - data_test$y)^2)
  
}

apply(error, 2, mean, na.rm=TRUE)

## Plot of variation explained for the last data set looked at in the simulation above
pca = prcomp(x)

plot(pca, main="Variation explained by each PC")
axis(1, at=seq(0.7, 11.5, length=10), labels=paste("PC", 1:10), las=2)



