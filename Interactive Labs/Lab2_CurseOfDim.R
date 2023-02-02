## Lab Week 2: The Curse of Dimensionality

# load libraries ----------------------------------------------------------

library(mvtnorm)
library(e1071)
library(class)



# find ratio of max dist to min dist --------------------------------------

## sample size and covariate dimension ranges
nVec = c(100,500,1000,3000)
pVec = c(1,3,5,10,15,20,25,30,40,50,75,100)

## store results
ratio = matrix(NA, length(pVec), length(nVec))

for (ni in 1 : length(nVec)) {
  for (pi in 1 : length(pVec)) {
    n = nVec[ni]
    p = pVec[pi]
    
    ## generate covariate matrix
    x = matrix(rnorm(n*p), n, p)
    
    ## get distances between all pairs of points
    distMat = as.matrix(dist(x))
    
    ## extract just upper triangle entries to avoid repeats
    distances = as.vector(distMat[which(upper.tri(distMat) == TRUE)])
    
    minDist = min(distances)
    maxDist = max(distances)
    
    ## store ratio of max to min
    ratio[pi,ni] = maxDist / minDist 
  }
}


# plot the results --------------------------------------------------------

## plot the effect of p on this ratio for different sample sizes
plot(pVec, ratio[,1], type="l", ylim=c(1,50), lwd=3,
     xlab = "# of covariates", ylab="Ratio")
lines(pVec, ratio[,2], lwd=3, col=2)
lines(pVec, ratio[,3], lwd=3, col=3)
lines(pVec, ratio[,4], lwd=3, col=4)
abline(h = 1, lty=2, lwd=2)
legend("topright", paste("n =", nVec), col=1:4, lwd=3, lty=1)

## can also look at the histogram of differences
n = 500
p = 5
x = matrix(rnorm(n*p), n, p)

## get distances between all pairs of points
distMat = as.matrix(dist(x))

## extract just upper triangle entries to avoid repeats
distances = as.vector(distMat[which(upper.tri(distMat) == TRUE)])

hist(distances)

## more predictors now
n = 500
p = 2500
x = matrix(rnorm(n*p), n, p)

## get distances between all pairs of points
distMat = as.matrix(dist(x))

## extract just upper triangle entries to avoid repeats
distances = as.vector(distMat[which(upper.tri(distMat) == TRUE)])

hist(distances)



# dimension reduction using variable selection -----------------------------------------------------

## now see what happens if we do dimension reduction
set.seed(2)
n = 1000
p = 200

x = matrix(rnorm(n*p), n, p)

## true regression function
f = function(x) {
  return(-1.5 + exp(x[,1]) + 2*x[,2] - log(x[,3]^2))
}

## generate outcome
y = rbinom(n, 1, p = pnorm(f(x)))

## generate testing data
xtest = matrix(rnorm(500*p), 500, p)
ytest = rbinom(n, 1, p = pnorm(f(xtest)))

## find the 5 covariates with largest correlation with y
keep = order(abs(cor(y, x)), decreasing=TRUE)[1:5]

x2 = x[,keep]
xtest2 = xtest[,keep]

## run knn on full set
knnMod = knn(train=x, test=xtest, k=10, cl=y)
knnPred = as.numeric(knnMod) - 1

## compare predictions with truth
mean(knnPred != ytest)

## now run only on the chosen covariates
knnMod5 = knn(train=x2, test=xtest2, k=10, cl=y)
knnPred5 = as.numeric(knnMod5) - 1

## compare predictions with truth
mean(knnPred5 != ytest)


## now run on the true set of covariates
x3  = x[,1:3]
xtest3 = xtest[,1:3]

knnModTrue = knn(train=x3, test=xtest3, k=10, cl=y)
knnPredTrue = as.numeric(knnModTrue) - 1

## compare predictions with truth
mean(knnPredTrue != ytest)

