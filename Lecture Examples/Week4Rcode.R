## Stock market analysis
library(ISLR)
library(MASS)

rm(list=ls())

data(Smarket)
names(Smarket)

## Pick days before 2004 to be training data
trainIndex = which(Smarket$Year < 2004)

SmarketTrain = Smarket[trainIndex,]
SmarketTest = Smarket[-trainIndex,]

mod = glm(Direction ~ . -Year -Today, 
          data=SmarketTrain, family=binomial)
summary(mod)

## predict test data given this model
testPred = 1*(predict(mod, newdata=SmarketTest, type="response") > 0.5)

## test set error
mean(testPred != (as.numeric(SmarketTest$Direction) - 1))


## Now pick random days to be training data
set.seed(4)
trainIndex = sample(1:nrow(Smarket), 750, replace=FALSE)

SmarketTrain = Smarket[trainIndex,]
SmarketTest = Smarket[-trainIndex,]

mod = glm(Direction ~ . -Year -Today + as.factor(Year), 
          data=SmarketTrain, family=binomial)
summary(mod)

## predict test data given this model
testPred = 1*(predict(mod, newdata=SmarketTest, type="response") > 0.5)

mean(testPred != (as.numeric(SmarketTest$Direction) - 1))


## Now look at LDA model

## First use model without year
modLDA = lda(Direction ~ . -Year -Today, 
             data=SmarketTrain)
modLDA

## predict test data given this model
testPredLDA = as.numeric(predict(modLDA, newdata=SmarketTest)$class) - 1

mean(testPredLDA != (as.numeric(SmarketTest$Direction) - 1))

## Now use the one that includes year as a dummy variable
modLDAyear = lda(Direction ~ . -Year -Today + as.factor(Year), 
                 data=SmarketTrain)
modLDAyear

## predict test data given this model
testPredLDAyear = as.numeric(predict(modLDAyear, 
                                     newdata=SmarketTest)$class) - 1

mean(testPredLDAyear != (as.numeric(SmarketTest$Direction) - 1))

## compare these predictions with the logistic regression ones
cor(testPredLDAyear, testPred)
table(testPredLDAyear, testPred)


## Now use QDA
modQDA = qda(Direction ~ . -Year -Today + as.factor(Year), 
             data=SmarketTrain)
modQDA

## predict test data given this model
testPredQDA = as.numeric(predict(modQDA, newdata=SmarketTest)$class) - 1

mean((testPredQDA != (as.numeric(SmarketTest$Direction) - 1)))


## Now try 100 different data sets of the same size
## and see how performance varies across data sets
nSim = 100
errorMat = matrix(NA, nSim, 4)

for (ni in 1 : nSim) {
  ## again specify the training and test data sets
  trainIndex = sample(1:nrow(Smarket), 750, replace=FALSE)
  
  SmarketTrain = Smarket[trainIndex,]
  SmarketTest = Smarket[-trainIndex,]
  
  ## fit all of the approaches as before and keep track of error rates
  mod = glm(Direction ~ . -Year -Today + as.factor(Year), 
            data=SmarketTrain, family=binomial)
  
  testPred = 1*(predict(mod, newdata=SmarketTest, type="response") > 0.5)
  
  errorMat[ni,1] = mean(testPred != (as.numeric(SmarketTest$Direction) - 1))
  
  ## LDA
  modLDAyear = lda(Direction ~ . -Year -Today + as.factor(Year), 
                   data=SmarketTrain)
  
  testPredLDAyear = as.numeric(predict(modLDAyear, 
                                       newdata=SmarketTest)$class) - 1
  
  errorMat[ni,2] = mean(testPredLDAyear != (as.numeric(SmarketTest$Direction) - 1))
  
  ## QDA
  modQDA = qda(Direction ~ . -Year -Today + as.factor(Year), 
               data=SmarketTrain)
  
  testPredQDA = as.numeric(predict(modQDA, newdata=SmarketTest)$class) - 1
  
  errorMat[ni,3] = mean((testPredQDA != (as.numeric(SmarketTest$Direction) - 1)))
  
  ## Coin flip estimator
  errorMat[ni,4] = mean((rbinom(500, 1, 0.5) != (as.numeric(SmarketTest$Direction) - 1)))
  
}

boxplot(errorMat, names=c("Logistic", "LDA", "QDA", "Coin flip"), main="Test error rates")

apply(errorMat, 2, mean)


