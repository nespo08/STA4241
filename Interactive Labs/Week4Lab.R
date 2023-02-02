####################################################
### Simulation study to compare LDA and Logistic ###
##### And assess robustness to LDA assumptions #####
####################################################
library(MASS)

n = 100
n_test = 10000

expit = function(x) {
  return(exp(x) / (1 + exp(x)))
}

## generate testing data sets for each of the five scenarios

set.seed(1)

## Normality
data_test_normal = data.frame(x = rnorm(n_test))
data_test_normal$y = rbinom(n_test, 1, p = expit(0.5*data_test_normal$x))

## T-distribution with 2 df
data_test_t = data.frame(x = rt(n_test, df=2))
data_test_t$y = rbinom(n_test, 1, p = expit(0.5*data_test_t$x))

## Gamma distribution (skewed)
data_test_gamma = data.frame(x = rgamma(n_test, 1,.1))
data_test_gamma$x = (data_test_gamma$x - mean(data_test_gamma$x))/
  sd(data_test_gamma$x)
data_test_gamma$y = rbinom(n_test, 1, p = expit(0.5*data_test_gamma$x))

## Categorical covariates
data_test_categorical = data.frame(x = sample(c(1,2,3,4), n_test, replace=TRUE))
data_test_categorical$y = rbinom(n_test, 1, 
                                 p = expit(-0.3 + 0.8*(data_test_categorical$x == 2) +
                                             0.4*(data_test_categorical$x == 3) +
                                             1.2*(data_test_categorical$x == 4)))

## Now generate data where the constant variance assumption is incorrect
data_test_var = data.frame(y = rbinom(n_test, 1, p=0.5))
data_test_var$x = rep(NA, n_test)

data_test_var$x[data_test_var$y == 1] = 
  rnorm(sum(data_test_var$y == 1), mean=1, sd=1)
data_test_var$x[data_test_var$y == 0] = 
  rnorm(sum(data_test_var$y == 0), sd=10)

## Data suitable for lda
data_test_lda = data.frame(y = rbinom(n_test, 1, p=0.3))
data_test_lda$x = rep(NA, n_test)

data_test_lda$x[data_test_lda$y == 1] = 
  rnorm(sum(data_test_lda$y == 1), mean=0.5)
data_test_lda$x[data_test_lda$y == 0] = 
  rnorm(sum(data_test_lda$y == 0), mean=0)

## Now we will generate many training data sets and see
## how well LDA and logistic regression predict the test data

## store results
nSim = 100
test_error_lda = matrix(NA, nSim, 6)
test_error_log = matrix(NA, nSim, 6)


for (ni in 1 : nSim) {
  ##############################################################
  #################### Normal covariates #######################
  ##############################################################
  
  ## Generate data
  data_normal = data.frame(x = rnorm(n))
  data_normal$y = rbinom(n, 1, p = expit(0.5*data_normal$x))
  
  ## fit logistic model and store error rate
  modLOG = glm(y ~ x, family=binomial, data=data_normal)
  pred_testLOG = 1*(predict(modLOG, data_test_normal, 
                            type="response") > 0.5)

  test_error_log[ni,1] = mean(pred_testLOG != data_test_normal$y)
  
  ## Fit LDA model and store error rate
  modLDA = lda(y ~ x, data=data_normal)
  pred_testLDA = as.numeric(predict(modLDA, newdata=
                                      data_test_normal)$class) - 1
  test_error_lda[ni,1] = mean(pred_testLDA != data_test_normal$y)


  ##############################################################
  ############### T-distribution covariates ####################
  ##############################################################
  
  ## Generate data
  data_t = data.frame(x = rt(n, df=2))
  data_t$y = rbinom(n, 1, p = expit(0.5*data_t$x))
  
  ## fit logistic model and store error rate
  modLOG = glm(y ~ x, family=binomial, data=data_t)
  pred_testLOG = 1*(predict(modLOG, data_test_t, 
                            type="response") > 0.5)
  test_error_log[ni,2] = mean(pred_testLOG != data_test_t$y)
  
  ## Fit LDA model and store error rate
  modLDA = lda(y ~ x, data=data_t)
  pred_testLDA = as.numeric(predict(modLDA, newdata=
                                      data_test_t)$class) - 1
  test_error_lda[ni,2] = mean(pred_testLDA != data_test_t$y)
  
  
  ##############################################################
  ##################### Gamma covariates #######################
  ##############################################################
  
  ## Generate data
  data_gamma = data.frame(x = rgamma(n, 1,.1))
  data_gamma$x = (data_gamma$x - mean(data_gamma$x))/
    sd(data_gamma$x)
  data_gamma$y = rbinom(n, 1, p = expit(0.5*data_gamma$x))
  
  ## fit logistic model and store error rate
  modLOG = glm(y ~ x, family=binomial, data=data_gamma)
  pred_testLOG = 1*(predict(modLOG, data_test_gamma, 
                            type="response") > 0.5)
  test_error_log[ni,3] = mean(pred_testLOG != data_test_gamma$y)
  
  ## Fit LDA model and store error rate
  modLDA = lda(y ~ x, data=data_gamma)
  pred_testLDA = as.numeric(predict(modLDA, newdata=
                                      data_test_gamma)$class) - 1
  test_error_lda[ni,3] = mean(pred_testLDA != data_test_gamma$y)
  
  
  ##############################################################
  #################### Categorical covariates #######################
  ##############################################################
  
  ## Generate data
  data_categorical = data.frame(x = sample(c(1,2,3,4), n, replace=TRUE))
  data_categorical$y = rbinom(n, 1, p = expit(-0.3 + 0.8*(data_categorical$x == 2) +
                                                0.4*(data_categorical$x == 3) +
                                                1.2*(data_categorical$x == 4)))

  ## fit logistic model and store error rate
  modLOG = glm(y ~ as.factor(x), family=binomial, data=data_categorical)
  pred_testLOG = 1*(predict(modLOG, data_test_categorical,
                            type="response") > 0.5)
  test_error_log[ni,4] = mean(pred_testLOG != data_test_categorical$y)

  ## Fit LDA model and store error rate
  modLDA = lda(y ~ as.factor(x), data=data_categorical)
  pred_testLDA = as.numeric(predict(modLDA, newdata=
                                      data_test_categorical)$class) - 1
  test_error_lda[ni,4] = mean(pred_testLDA != data_test_categorical$y)
  
  
  ##############################################################
  #################### Unequal variances #######################
  ##############################################################
  
  ## Generate data
  data_var = data.frame(y = rbinom(n, 1, p=0.5))
  data_var$x = rep(NA, n)
  
  data_var$x[data_var$y == 1] = 
    rnorm(sum(data_var$y == 1), mean=1, sd=1)
  data_var$x[data_var$y == 0] = 
    rnorm(sum(data_var$y == 0), sd=10)
  
  ## fit logistic model and store error rate
  modLOG = glm(y ~ x, family=binomial, data=data_var)
  pred_testLOG = 1*(predict(modLOG, data_test_var, 
                            type="response") > 0.5)
  test_error_log[ni,5] = mean(pred_testLOG != data_test_var$y)
  
  ## Fit LDA model and store error rate
  modLDA = lda(y ~ x, data=data_var)
  pred_testLDA = as.numeric(predict(modLDA, newdata=
                                      data_test_var)$class) - 1
  test_error_lda[ni,5] = mean(pred_testLDA != data_test_var$y)
  
  
  ##############################################################
  ################### LDA suitable data ########################
  ##############################################################
  
  data_lda = data.frame(y = rbinom(n, 1, p=0.3))
  data_lda$x[data_lda$y == 1] = 
    rnorm(sum(data_lda$y == 1), mean=0.5)
  data_lda$x[data_lda$y == 0] = 
    rnorm(sum(data_lda$y == 0), mean=0)
  
  ## fit logistic model and store error rate
  modLOG = glm(y ~ x, family=binomial, data=data_lda)
  pred_testLOG = 1*(predict(modLOG, data_test_lda, 
                            type="response") > 0.5)
  test_error_log[ni,6] = mean(pred_testLOG != data_test_lda$y)
  
  ## Fit LDA model and store error rate
  modLDA = lda(y ~ x, data=data_lda)
  pred_testLDA = as.numeric(predict(modLDA, newdata=
                                      data_test_lda)$class) - 1
  test_error_lda[ni,6] = mean(pred_testLDA != data_test_lda$y)
}

## Plot the results
par(mfrow=c(3,2), pty='s', mar=c(2,2,2,2))
boxplot(x=cbind(test_error_lda[,1], test_error_log[,1]),
        names = c("LDA", "Logistic"), main="Normal")

boxplot(x=cbind(test_error_lda[,2], test_error_log[,2]),
        names = c("LDA", "Logistic"), main="T-dist")

boxplot(x=cbind(test_error_lda[,3], test_error_log[,3]),
        names = c("LDA", "Logistic"), main="Gamma")

boxplot(x=cbind(test_error_lda[,4], test_error_log[,4]),
        names = c("LDA", "Logistic"), main="Categorical covariates")

boxplot(x=cbind(test_error_lda[,5], test_error_log[,5]),
        names = c("LDA", "Logistic"), main="Nonequal variances")

boxplot(x=cbind(test_error_lda[,6], test_error_log[,6]),
        names = c("LDA", "Logistic"), main="LDA data")








####################################################
### Creating a data set with perfect separation ####
##### What happens to both LDA and Logistic reg ####
####################################################

## remove the 3 by 2 plotting window
dev.off()

## sample size
n = 200

## create an outcome that is perfectly separated by x
data = data.frame(x=rnorm(n))
data$y = 1*(data$x > 0)

## look at testing data on a grid
data_test = data.frame(x = seq(-3, 3, length=1000))
data_test$y = 1*(data_test$x > 0) 

## fit GLM and look at estimated coefficients!
modGLM = glm(y ~ x, family=binomial, data=data)
summary(modGLM)

plot(data$x,data$y)

## predicted probability of Y=1 on the grid
lines(data_test$x, expit(modGLM$coefficients[1] + 
                      modGLM$coefficients[2]*data_test$x), type='l', lwd=3)

## Fit LDA model
modLDA = lda(y ~ x, data=data)
modLDA


## Compare predictions
predGLM = 1*(predict(modGLM, data_test, 
                     type="response") > 0.5)
predLDA = as.numeric(predict(modLDA, newdata=
                               data_test)$class) - 1

## testing error rates
mean(predGLM != data_test$y)
mean(predLDA != data_test$y)

