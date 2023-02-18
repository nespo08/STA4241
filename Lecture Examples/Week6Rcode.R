rm(list=ls())

library(mvtnorm)
library(e1071)
library(splines)
library(MASS)
library(ISLR)
library(xtable)
library(class)

## Load in the auto data
data(Auto)

## Check error rates by splitting the data in half and trying
## Polynomial regression for degrees up to 10

nSim = 20
D = 10
testError = matrix(NA, nSim, D)

for (ii in 1 : nSim) {
  ## Split data into training/testing
  samp = sample(1:nrow(Auto), 196, replace=FALSE) # 50 / 50, no replacement for CV
  trainAuto = Auto[samp,]
  testAuto = Auto[-samp,]
  
  ## Try all possible degrees
  for (dd in 1 : D) {
    mod = lm(mpg ~ poly(horsepower, dd), data=trainAuto)
    predTest = as.numeric(predict(mod, testAuto))  
    testError[ii,dd] = mean((predTest - testAuto$mpg)^2)
  }
}

## Plot all of the results
plot(1:D, testError[1,], type='l', lwd=2, xlab="Degree of polynomial", ylab="Testing MSE",
     ylim=c(0,40), main="Different validation sets")
for (ii in 2 : nSim) {
  lines(1:D, testError[ii,], lwd=2, col=ii)
}

## For each of the 20 data splits, which degree of polynomial is chosen (the min of each iteration)
apply(testError, 1, which.min)

rm(list=ls())

## Load in the stock market data
data(Smarket)

## Run 10-fold cross validation on stock market data to evaluate all approaches
## Note this part will take a few minutes to run
K = 10

## groups is a vector of which CV group each observation is in for 10-fold CV
groups <- cut(1:nrow(Smarket), breaks = 10, labels = F)

## Need to make year variable a factor so R knows not to treat it as numeric
Smarket$Year = factor(Smarket$Year)

## Keep track of the error for each approach
errorMat = matrix(NA, K, 10)

## May take a few minutes to run!
for (k in 1 : K) {
  ## Split the data into training/testing
  testIndex = which(groups == k)
  
  SmarketTrain = Smarket[-testIndex,]
  SmarketTest = Smarket[testIndex,]
  
  print(k)
  
  ## First fit the GLM
  mod = glm(Direction ~ . -Today, 
            data=SmarketTrain, family=binomial)
  
  testPred = 1*(predict(mod, newdata=SmarketTest, type="response") > 0.5) # More important
  trainPred = 1*(predict(mod, newdata=SmarketTrain, type="response") > 0.5)
  
  errorMat[k,1] = mean(testPred != (as.numeric(SmarketTest$Direction) - 1))
  
  ## Now use LDA
  modLDAyear = lda(Direction ~ . -Today, 
                   data=SmarketTrain)
  
  testPredLDAyear = as.numeric(predict(modLDAyear, 
                                       newdata=SmarketTest)$class) - 1
  
  errorMat[k,2] = mean(testPredLDAyear != (as.numeric(SmarketTest$Direction) - 1))
  
  ## QDA
  modQDA = qda(Direction ~ . -Today, 
               data=SmarketTrain)
  
  testPredQDA = as.numeric(predict(modQDA, newdata=SmarketTest)$class) - 1
  
  errorMat[k,3] = mean((testPredQDA - (as.numeric(SmarketTest$Direction) - 1))^2)
  
  ## Radial SVM
  tune.svm = tune(svm, Direction ~. -Today, 
                  data=SmarketTrain, kernel="radial",
                  ranges=list(cost=c(0.01, 1, 5,10, 100),
                              gamma=c(0.001, 0.01, 0.1, 1)))
  fit = tune.svm$best.model
  
  predSVM = 1*(as.character(predict(fit, SmarketTest)) == "Up")
  
  errorMat[k,4] = mean(predSVM != (as.numeric(SmarketTest$Direction) - 1))
  
  ## Polynomial SVM
  tune.svm = tune(svm, Direction ~. -Today, 
                  data=SmarketTrain, kernel="polynomial",
                  ranges=list(cost=c(0.01, 1, 5,10, 100),
                              degree=c(1,2,3,4)))
  fit = tune.svm$best.model
  
  predSVM = 1*(as.character(predict(fit, SmarketTest)) == "Up")
  
  errorMat[k,5] = mean(predSVM != (as.numeric(SmarketTest$Direction) - 1))
  
  ## Try KNN for a few different K values
  knnTrainY = SmarketTrain$Direction
  knnTestY = SmarketTest$Direction
  
  knnDat = SmarketTrain
  knnDat$Today = NULL
  knnDat$Direction = NULL
  
  knnTestDat = SmarketTest
  knnTestDat$Today = NULL
  knnTestDat$Direction = NULL
  
  knnPred5 = 1*(as.character(knn(train=knnDat, 
                                 test=knnTestDat, cl=knnTrainY, k=5)) == "Up") 
  knnPred10 = 1*(as.character(knn(train=knnDat, 
                                  test=knnTestDat, cl=knnTrainY, k=10)) == "Up") 
  knnPred20 = 1*(as.character(knn(train=knnDat, 
                                  test=knnTestDat, cl=knnTrainY, k=20)) == "Up") 
  knnPred50 = 1*(as.character(knn(train=knnDat, 
                                  test=knnTestDat, cl=knnTrainY, k=50)) == "Up") 
  knnPred200 = 1*(as.character(knn(train=knnDat, 
                                   test=knnTestDat, cl=knnTrainY, k=200)) == "Up") 
  
  errorMat[k,6] = mean(knnPred5 != (as.numeric(SmarketTest$Direction) - 1))
  errorMat[k,7] = mean(knnPred10 != (as.numeric(SmarketTest$Direction) - 1))
  errorMat[k,8] = mean(knnPred20 != (as.numeric(SmarketTest$Direction) - 1))
  errorMat[k,9] = mean(knnPred50 != (as.numeric(SmarketTest$Direction) - 1))
  errorMat[k,10] = mean(knnPred200 != (as.numeric(SmarketTest$Direction) - 1))
  
}

## Look at the average error across the K volds
apply(errorMat, 2, mean, na.rm=TRUE)

## Plot the data
names = c("Logistic", "LDA", "QDA", "SVM radial", "SVM polynomial",
          paste("KNN, K=", c(5,10,20,50,200), sep=""))

par(mar=c(8,4,2,2))
barplot(apply(errorMat, 2, mean, na.rm=TRUE), names.arg = names, las=2,
        main="10-fold CV estimates")
dev.off()






## Bootstrap the simple example on means of a normal random variable
n = 20
set.seed(1)
x = rnorm(n, mean=10)

## Keep track of 1000 bootstrap estimates of the mean
nBoot = 1000
estBoot = rep(NA, nBoot)

for (nb in 1 : nBoot) {
  ## randomly choose indices to make bootstrap sample
  samp = sample(1:n, n, replace=TRUE) # We want repeats for bootstrapping 
  xBoot = x[samp]
  estBoot[nb] = mean(xBoot) # Sample mean (theta_hat in this case)
}

hist(estBoot, main="Bootstrap confidence interval", xlab="Estimates")
abline(v = quantile(estBoot, c(0.025, 0.975)), col=2, lwd=2)


## Now do the bootstrap for the linear regression example
## Where we want to estimate sigma squared

## First let's get the sampling distribution by repeatedly sampling
## From the known data generating process. Normally this is unknown
## and we can't do this, but we can here because it is simulated data

estSigma = rep(NA, 10000)
for (i in 1 : 10000) {
  n = 30
  x1 = rnorm(n)
  x2 = rnorm(n)
  
  y = x1 + 0.5*x2 + rnorm(n, sd=2)
  
  linMod = lm(y ~ x1 + x2)
  
  estSigma[i] = sum((y - linMod$fitted.values)^2 / (n - 2 - 1))
}

## Sampling distribution
hist(estSigma)

## Now approximate this with the bootstrap on one data set
n = 30
set.seed(5)
x1 = rnorm(n)
x2 = rnorm(n)

y = x1 + 0.5*x2 + rnorm(n, sd=2)

## Estimate sigma squared for each bootstrap iteration
nBoot = 1000
estSigmaBoot = rep(NA, nBoot)
for (nb in 1 : nBoot) {
  samp = sample(1:n, n, replace=TRUE)
  yBoot = y[samp] # Generate dataset with new X1, X2, and Y
  x1Boot = x1[samp]
  x2Boot = x2[samp]
  
  linModBoot = lm(yBoot ~ x1Boot + x2Boot)
  
  estSigmaBoot[nb] = sum((yBoot - linModBoot$fitted.values)^2 / (n - 2 - 1))
}

## Plot bootstrap distribution
hist(estSigmaBoot, probability=TRUE, xlab="Residual variance", xlim=c(0,7),
     main="Bootstrap distribution")

## Plot the true sampling distribution we got before
densTrue = density(estSigma)
lines(densTrue$x, densTrue$y, lwd=2)


rm(list=ls())

## Let's do the bootstrap on the auto data
data(Auto)

## Get estimates and standard errors (analytic)
linMod = lm(mpg ~ horsepower, data=Auto)
summary(linMod)

## Now bootstrap to get them
nBoot = 10000
estBoot = matrix(NA, nBoot, 2)

for (nb in 1 : nBoot) {
  samp = sample(1:nrow(Auto), nrow(Auto), replace=TRUE)
  autoBoot = Auto[samp,]
  
  linModBoot = lm(mpg ~ horsepower, data=autoBoot)
  
  estBoot[nb,] = linModBoot$coefficients
}

## Bootstrap standard errors
apply(estBoot, 2, sd)

## Now fit the quadratic model

linMod = lm(mpg ~ horsepower + I(horsepower^2), data=Auto)
summary(linMod)

## Again get bootstrap estimates of standard errors
nBoot = 10000
estBoot = matrix(NA, nBoot, 3)

for (nb in 1 : nBoot) {
  samp = sample(1:nrow(Auto), nrow(Auto), replace=TRUE)
  autoBoot = Auto[samp,]
  
  linModBoot = lm(mpg ~ horsepower + I(horsepower^2), data=autoBoot)
  
  estBoot[nb,] = linModBoot$coefficients
}

## Bootstrap standard errors
apply(estBoot, 2, sd)


## Now try the parametric bootstrap
estBoot = matrix(NA, nBoot, 3)

## First get the coefficient estimates from the original data
linMod = lm(mpg ~ horsepower + I(horsepower^2), data=Auto)
coefEst = linMod$coefficients
sdEst = sqrt(sum(linMod$residuals^2)/(nrow(Auto)-2-1))

## Parametric bootstrap
for (nb in 1 : nBoot) {
  ## Create a new data set where we generate a new outcome
  ## From the fitted model above
  autoBoot = data.frame(horsepower = Auto$horsepower,
                        mpg = coefEst[1] + coefEst[2]*Auto$horsepower +
                          coefEst[3]*Auto$horsepower^2 + 
                          rnorm(nrow(Auto), sd=sdEst))
  
  linModBoot = lm(mpg ~ horsepower + I(horsepower^2), data=autoBoot)
  
  estBoot[nb,] = linModBoot$coefficients
}

## Standard errors from parameteric bootstrap
apply(estBoot, 2, sd)

