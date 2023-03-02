# Load libraries
library(tidyverse)
library(class)
library(ggplot2)
library(gridExtra)
library(MASS)
library(knitr)
library(reshape2)
library(e1071)
library(caret)

set.seed(2)

# Read in data
q1 <- read.csv("Homeworks/Data/HW3/Crabs.csv")
q1_data <- data.frame(q1)
weight <- q1_data$weight
width <- q1_data$width
color <- q1_data$color
spine <- q1_data$spine
q1_y <- q1_data$y
q1_n <- nrow(q1_data)

# Split data into training/testing
samp <- sample(1:q1_n, floor(q1_n * 0.8), replace = FALSE) # 80 / 20, no replacement
q1_train <- q1_data[samp,]
q1_test <- q1_data[-samp,]

# Keep track of the error for each approach
numTrials <- 100
errorMat <- matrix(NA, numTrials, 6)

# Tune KNN and both SVMs with previous training data

# KNN
trctrl <- trainControl(method = "repeatedcv", number = 10)
q1_knn_tune <- train(as.factor(y) ~ .,
                     data = q1_train,
                     method = "knn",
                     trControl = trctrl,
                     tuneLength = 10)
q1_knn_param <- q1_knn_tune$bestTune$k

# SVM - Polynomial
q1_svm_poly <- tune(svm,
                    as.factor(y) ~ .,
                    data = q1_train,
                    kernel = "polynomial",
                    ranges=list(cost=c(0.01, 1, 5, 10, 100),
                                degree=c(1,2))) 
q1_svm_poly_param <- q1_svm_poly$best.parameters


# SVM - Radial
q1_svm_rad <- tune(svm,
                   as.factor(y) ~ .,
                   data = q1_train,
                   kernel = "radial",
                   ranges=list(cost=c(0.01, 1, 5, 10, 100),
                               gamma=c(0.001, 0.01, 0.1, 1))) 
q1_svm_rad_param <- q1_svm_rad$best.parameters

for(n in 1:numTrials){
  
  # Pick 25 random observations to be testing data
  sampIndex <- sample(1:q1_n, 25, replace = FALSE)
  
  q1_iii_test <- q1_data[sampIndex,]
  q1_iii_train <- q1_data[-sampIndex,]
  
  # Logistic regression
  q1_logit <- glm(y ~ ., 
                  data = q1_iii_train,
                  family = binomial)
  
  test_pred_logit <- 1*(predict(q1_logit, newdata = q1_iii_test, type="response") > 0.5)
  
  errorMat[n,1] <- mean(test_pred_logit != (q1_iii_test$y))
  
  # LDA
  q1_lda <- lda(y ~ ., 
                data = q1_iii_train)
  
  test_pred_lda <- as.numeric(predict(q1_lda, 
                                      newdata = q1_iii_test)$class) - 1
  
  errorMat[n,2] <- mean(test_pred_lda != (q1_iii_test$y))
  
  # QDA
  q1_qda <- qda(y ~ ., 
                data = q1_iii_train)
  
  test_pred_qda <- as.numeric(predict(q1_qda, 
                                      newdata = q1_iii_test)$class) - 1
  errorMat[n,3] <- mean(test_pred_qda != (q1_iii_test$y))
  
  # KNN
  #trctrl <- trainControl(method = "repeatedcv", number = 10)
  # q1_knn <- train(y ~ .,
  #                 data = q1_iii_train,
  #                 method = "knn",
  #                 trControl = trctrl,
  #                 tuneLength = 10)
  
  # Store training/testing y values and set them to NULL is training/testing data set
  knn_iii_train <- q1_iii_train
  knn_iii_test <- q1_iii_test
  
  q1_knn_train_y <- knn_iii_train$y
  q1_knn_test_y <- knn_iii_test$y
  
  knn_iii_train$y <- NULL
  knn_iii_test$y <- NULL
  
  q1_knn_fit <- knn(train = q1_iii_train, 
                    test = q1_iii_test, cl = q1_knn_train_y, k = q1_knn_param) # Run knn
  
  # test_pred_knn <- 1*((predict(q1_knn_fit, newdata = q1_iii_test) > 0.5)) NOT NEEDED
  
  errorMat[n,4] <- mean(q1_knn_fit != (q1_iii_test$y))
  
  # SVMs w/ Polynomial Kernel
  # q1_svm_poly <- tune(svm,
  #                     y ~ .,
  #                 data = q1_iii_train,
  #                 kernel = "polynomial",
  #                 ranges=list(cost=c(0.01, 1, 5, 10, 100),
  #                             gamma=c(0.001, 0.01, 0.1, 1)))
  
  # NEED TO INCLUDE y
  q1_svm_poly_fit <- svm(as.factor(y) ~ ., 
                         data = q1_iii_train,
                         kernel = "polynomial",
                         cost = q1_svm_poly_param$cost,
                         scale = F,
                         degree = q1_svm_poly_param$degree)
  
  test_pred_svm_poly <- (predict(q1_svm_poly_fit, newdata = q1_iii_test))
  
  errorMat[n,5] <- mean(test_pred_svm_poly != (q1_iii_test$y))
  
  # SVMs w/ Radial Kernel
  # q1_svm_rad <- tune(svm,
  #                     y ~ .,
  #                 data = q1_iii_train,
  #                 kernel = "radial",
  #                 ranges=list(cost=c(0.01, 1, 5, 10, 100),
  #                             gamma=c(0.001, 0.01, 0.1, 1)))
  
  q1_svm_rad_fit <- svm(as.factor(y) ~ ., 
                        data = q1_iii_train,
                        kernel = "radial",
                        cost = q1_svm_rad_param$cost,
                        scale = F,
                        gamma = q1_svm_rad_param$gamma)
  
  test_pred_svm_rad <- (predict(q1_svm_rad_fit, newdata = q1_iii_test))
  
  errorMat[n,6] <- mean(test_pred_svm_rad != (q1_iii_test$y))
}

# Find mean error rate of each method
mean_logit <- mean(errorMat[,1])
mean_lda <- mean(errorMat[,2])
mean_qda <- mean(errorMat[,3])
mean_knn <- mean(errorMat[,4])
mean_svm_poly <- mean(errorMat[,5])
mean_svm_rad <- mean(errorMat[,6])

# Print values
q1_out <- cbind(mean_logit, mean_lda, mean_qda, mean_knn, mean_svm_poly, mean_svm_rad)
colnames(q1_out) <- c("Logistic", "LDA", "QDA", "KNN", "SVM-P", "SVM-R")
kable(q1_out, digits = 4, align = "l")

# Boxplot
apply(errorMat, 2, mean, na.rm = TRUE)

boxplot(errorMat, names = c("Logistic", "LDA", "QDA", "KNN", "SVM-P", "SVM-R"), 
        main = "Comparing Error Rates")