# STA4241 Midterm - Nicholas Esposito

# Load all libraries and set seed ------------------------------------------------------

library(tidyverse)
library(ggplot2)
library(gridExtra)
library(knitr)
library(MASS) # For QDA and LDA
library(e1071) # For SVM
library(dplyr) # Renaming dataframe columns
library(reshape2) # Reorganize corr. matrix in Q4
library(caret) # Used for tuning PCR, PLS
library(glmnet) # Used for Lasso/Ridge Regression
library(pls) # Used for PLS and PCR
library(leaps) # Used for variable selection
library(ggcorrplot) # Used to rearrange correlation heatmap

set.seed(2)

# Question 1 --------------------------------------------------------------

## i. Propose an unbiased estimator of α and show that it is unbiased for estimating α.

# Answer in markdown

## ii. Find the sampling distribution of the estimator

# Visualize the sampling distribution of our estimator
q1_visualize_dist <- function(n){
  q1_xbar <- c()
  q1_x2bar <- c()
  for (i in 1:1000){
    # Draw x values from Gamma dist. and find xbar and x^2
    q1_x <- rgamma(n, 1, 0.01)
    q1_xbar[i] <- mean(q1_x)
    q1_x2bar[i] <- mean(q1_x^2)
  } 
  
  # Generate estimates of alphahat
  q1_alphahat <- data.frame(q1_xbar^2 / (q1_x2bar - (q1_xbar^2)))
  colnames(q1_alphahat) <- c("Estimates")
  
  # Plot distribution of the estimates
  q1_plot <- q1_alphahat %>% 
    ggplot(aes(x = Estimates)) +
    geom_histogram(aes(y = ..density..),
                   color = "black",
                   fill = "white") +
    stat_function(fun = dnorm,
                  args = list(mean = mean(q1_alphahat$Estimates),
                              sd = sd(q1_alphahat$Estimates))) +
    labs(title = paste("n = ", as.character(n)))
  
  return(q1_plot)
}

# Find distribution at n = 10, 1000, 10000
q1_plot_10 <- q1_visualize_dist(10)
q1_plot_1000 <- q1_visualize_dist(1000)
q1_plot_10000 <- q1_visualize_dist(10000)

grid.arrange(q1_plot_10, q1_plot_1000, q1_plot_10000, ncol=2)

## iii. Describe how to implement a parametric bootstrap for this particular problem.

# Answer in markdown

## iv. Find the variance using the proposed bootstrap (Gamma dist.)
q1_boot_xbar <- c()
q1_boot_x2bar <- c()
q1_n <- 1000
q1_B <- 1000

for (boot in 1:q1_B){
  # Draw x values from Gamma dist. and find xbar and x^2
  q1_boot_x <- rgamma(1000, 1, 0.01)
  q1_boot_xbar[boot] <- mean(q1_boot_x)
  q1_boot_x2bar[boot] <- mean(q1_boot_x^2)
} 

# Generate estimates of alphahat
q1_boot_alphahat <- data.frame(q1_boot_xbar^2 / (q1_boot_x2bar - (q1_boot_xbar^2)))
colnames(q1_boot_alphahat) <- c("Estimates")

# Variance of the bootstrap estimates
q1_boot_var <- sd(q1_boot_alphahat$Estimates)^2

# Print values
q1_iv_out <- cbind(q1_boot_var)
colnames(q1_iv_out) <- c("Variance Using Parametric Bootstrap in (iii)")
kable(q1_iv_out, digits = 4, align = "l")

## v. What is the expected value of your answer in (iv)?

# Answer in markdown

## vi. Describe how to implement a parametric bootstrap for this particular problem when assuming a normal distribution for the data. 

# Answer in markdown

## vii. Find the expected value of the variance using the proposed bootstrap (Normal dist.)

q1_norm_boot_xbar <- c()
q1_norm_boot_x2bar <- c()

# Find mean and sd of x values from Gamma dist.
q1_x_gamma <- rgamma(q1_n, 1, 0.01)
q1_mean <- mean(q1_x_gamma)
q1_sd <- sd(q1_x_gamma)

for (boot in 1:q1_B){
  # Draw x values from Normal dist. and find xbar and x^2
  q1_norm_boot_x <- rnorm(1000, q1_mean, q1_sd)
  q1_norm_boot_xbar[boot] <- mean(q1_norm_boot_x)
  q1_norm_boot_x2bar[boot] <- mean(q1_norm_boot_x^2)
} 

# Generate estimates of alphahat
q1_norm_boot_alphahat <- data.frame(q1_norm_boot_xbar^2 / (q1_norm_boot_x2bar - (q1_norm_boot_xbar^2)))
colnames(q1_norm_boot_alphahat) <- c("Estimates")

# Variance of the bootstrap estimates
q1_norm_boot_var <- sd(q1_norm_boot_alphahat$Estimates)^2

# Since the variance is a constant, the expected value will be that constant

# Print values
q1_vii_out <- cbind(q1_norm_boot_var)
colnames(q1_vii_out) <- c("Expected Value of the Variance Using Parametric Bootstrap in (vi)")
kable(q1_vii_out, digits = 4, align = "l")

# viii. Given your answer to the previous question, do you think this misspecified version of the bootstrap will perform well?

# Answer in markdown

# Question 2 --------------------------------------------------------------

## i. Assess the performance of three different bootstraps for performing inference of alpha: The two parametric bootstraps from Q1 and the standard nonparam. bootstrap

# Function for estimation of alpha
estAlpha <- function(bootVals){
  mean_sq <- mean(bootVals)^2 # Evaluate estimator in parts
  x_sq <- (sum((bootVals)^2)) / (length(bootVals))
  
  est_alpha_val <- (mean_sq) / (x_sq - mean_sq)
  return(est_alpha_val)
}

# Perform 1000 simulation trials
numTrials <- 1000
q2_nonp_cov <- matrix(NA, numTrials, 2) # 3 bootstrap variations
q2_p_gamma_cov <- matrix(NA, numTrials, 2)
q2_p_norm_cov <- matrix(NA, numTrials, 2)
q2_B <- 1000 # 1000 bootstraps

# Data set information
q2_n <- 10
q2_alpha <- 1
q2_beta <- 0.01

# Store bootstrap values for visualization
q2_nonp_viz <- c()
q2_p_gamma_viz <- c()
q2_p_norm_viz <- c()

for(trial in 1:numTrials){
  
  # Generate data set
  q2_x <- rgamma(q2_n, shape = q2_alpha, scale = q2_beta)
  
  # --- Non-parametric bootstrapping ---
  
  # Keep track of B = 1000 bootstrap estimates 
  q2_nonp_bootstrap_vals <- rep(NA, q2_B)
  
  for(boot in 1:q2_B){
    # Find random sample from sample data to perform bootstrapping
    q2_nonp_boot <- sample(q2_x, q2_n, replace = TRUE)
    q2_nonp_bootstrap_vals[boot] <- estAlpha(q2_nonp_boot) # Estimator for alpha
  }
  
  # Use percentile method
  q2_nonp_cov[trial,] <- quantile(q2_nonp_bootstrap_vals, c(0.025, 0.975)) # q2_nonp_cov[trial,1] stores q0.025,  q2_nonp_cov[trial,1] stores q0.975
  
  # Store non-parametric bootstrap values for visualization
  q2_nonp_viz <- append(q2_nonp_viz, q2_nonp_bootstrap_vals)
  
  # --- Parametric bootstrapping ---
  
  # Keep track of B = 1000 bootstrap estimates 
  q2_p_gamma_bootstrap_vals <- rep(NA, q2_B)
  q2_p_norm_bootstrap_vals <- rep(NA, q2_B)
  
  # Estimate using estimator from Q1
  q2_est_alpha <- estAlpha(q2_x)
  
  # Find mean and sd of x values
  q2_mean <- mean(q2_x)
  q2_sd <- sd(q2_x)
  
  for(boot in 1:q2_B){
    # Generate data from the distribution our sample follows
    
    # 1. Gamma distribution
    q2_p_gamma_bootstrap_vals[boot] <- estAlpha(rgamma(q2_n, q2_est_alpha, q2_beta)) # Estimator for alpha
    
    # 2. Normal distribution
    q2_p_norm_bootstrap_vals[boot] <- estAlpha(rnorm(q2_n, q2_mean, q2_sd)) # Estimator for alpha
  }
  
  # Use percentile method - gamma dist.
  q2_p_gamma_cov[trial,] <- quantile(q2_p_gamma_bootstrap_vals, c(0.025, 0.975))
  q2_p_norm_cov[trial,] <- quantile(q2_p_norm_bootstrap_vals, c(0.025, 0.975))
  
  # Store parametric bootstrap values for visualization
  q2_p_gamma_viz <- append(q2_p_gamma_viz, q2_p_gamma_bootstrap_vals)
  q2_p_norm_viz <- append(q2_p_norm_viz, q2_p_norm_bootstrap_vals)
}

# Non-parametric coverage rate
q2_nonp_cov_rate <- mean(1*(q2_nonp_cov[,1] < q2_alpha & q2_nonp_cov[,2] > q2_alpha))

# Parametric coverage rate - gamma dist.
q2_p_gamma_cov_rate <- mean(1*(q2_p_gamma_cov[,1] < q2_alpha & q2_p_gamma_cov[,2] > q2_alpha))

# Parametric coverage rate - normal dist.
q2_p_norm_cov_rate <- mean(1*(q2_p_norm_cov[,1] < q2_alpha & q2_p_norm_cov[,2] > q2_alpha))

# Print values
q2_i_out <- cbind(q2_nonp_cov_rate, q2_p_gamma_cov_rate, q2_p_norm_cov_rate)
colnames(q2_i_out) <- c("Non-Parametric", "Parametric - Gamma Dist.", "Parametric - Normal Dist.")
kable(q2_i_out, digits = 4, align = "l")

## ii. Visualize the sampling distribution of alpha^

# Make each vector of est. values a dataframe
q2_nonp_df <- data.frame(q2_nonp_viz)
colnames(q2_nonp_df) <- c("Estimate")

q2_p_gamma_df <- data.frame(q2_p_gamma_viz)
colnames(q2_p_gamma_df) <- c("Estimate")

q2_p_norm_df <- data.frame(q2_p_norm_viz)
colnames(q2_p_norm_df) <- c("Estimate")

# Plot the distribution of each bootstrap's estimates
q2_nonp_hist <-  q2_nonp_df %>%
  ggplot(aes(x = Estimate)) +
  geom_density(color = "black",
                 fill = "blue",
                 binwidth = 0.25) +
  labs(title = "Non-Parametric Bootstrap Distribution") +
  xlim(0,10) +
  ylim(0,1) +
  geom_vline(xintercept = q2_alpha,
             linetype = "dashed",
             color = "red",
             size = 0.5)

q2_p_gamma_hist <- q2_p_gamma_df %>%
  ggplot(aes(x = Estimate)) +
  geom_density(color = "black",
                 fill = "blue") +
  labs(title = "Parametric Bootstrap Distribution - Gamma Dist.") +
  xlim(0,10) +
  ylim(0,1) +
  geom_vline(xintercept = q2_alpha,
             linetype = "dashed",
             color = "red",
             size = 0.5)

q2_p_norm_hist <-  q2_p_norm_df %>%
  ggplot(aes(x = Estimate)) +
  geom_density(color = "black",
                 fill = "blue") +
  labs(title = "Parametric Bootstrap Distribution - Normal Dist.") +
  xlim(0,10) +
  ylim(0,1) +
  geom_vline(xintercept = q2_alpha,
             linetype = "dashed",
             color = "red",
             size = 0.5)

grid.arrange(q2_nonp_hist, q2_p_gamma_hist, q2_p_norm_hist, ncol=2)

## iii. Explain the results of each bootstrap approach.

# Answer in markdown

## iv. Does the performance of the parametric bootstrap that incorrectly assumes a normal distribution for Xi depend in any way on the true values for alpha and beta?

norm_dist_sim <- function(alphaVal, betaVal){
  # Store conf. intervals
  q2_iv_p_norm_cov <- matrix(NA, numTrials, 2)
  
  # Run 1000 trials with 1000 bootstraps for varying values of alpha and beta (same num. of trials and bootstraps as above)
  for(trial in 1:numTrials){
    
    # Generate data set
    q2_iv_x <- rgamma(q2_n, shape = alphaVal, scale = betaVal)
    
    # Keep track of B = 1000 bootstrap estimates 
    q2_iv_p_norm_bootstrap_vals <- rep(NA, q2_B)
    
    # Find mean and sd of x values
    q2_iv_mean <- mean(q2_iv_x)
    q2_iv_sd <- sd(q2_iv_x)
    
    for(boot in 1:q2_B){
      # Generate data from the Normal distribution
      q2_iv_p_norm_bootstrap_vals[boot] <- estAlpha(rnorm(q2_n, q2_iv_mean, q2_iv_sd)) # Estimator for alpha
    }
    
    # Use percentile method - gamma dist.
    q2_iv_p_norm_cov[trial,] <- quantile(q2_iv_p_norm_bootstrap_vals, c(0.025, 0.975))
  }
  
  return(mean(1*(q2_iv_p_norm_cov[,1] < alphaVal & q2_iv_p_norm_cov[,2] > alphaVal)))
}

# Alpha: 2, Beta: 0.02
q2_iv_alpha2_beta2 <- norm_dist_sim(2, 0.02)

# Alpha: 2, Beta: 0.05
q2_iv_alpha2_beta5 <- norm_dist_sim(2, 0.05)

# Alpha: 5, Beta: 0.02
q2_iv_alpha5_beta2 <- norm_dist_sim(5, 0.02)

# Alpha: 5, Beta: 0.05
q2_iv_alpha5_beta5 <- norm_dist_sim(5, 0.05)

# Alpha: 10, Beta: 0.1
q2_iv_alpha10_beta1 <- norm_dist_sim(10, 0.1)


# Print values
q2_iv_out <- cbind(q2_p_norm_cov_rate, q2_iv_alpha2_beta2, q2_iv_alpha2_beta5, q2_iv_alpha5_beta2, q2_iv_alpha5_beta5, q2_iv_alpha10_beta1)
colnames(q2_iv_out) <- c("Alpha: 1, Beta: 0.01", "Alpha: 2, Beta: 0.02", "Alpha: 2, Beta: 0.05", "Alpha: 5, Beta: 0.02", "Alpha: 5, Beta: 0.05", "Alpha: 10, Beta: 0.1")
kable(q2_iv_out, digits = 4, align = "l")

# Answer in markdown

# Question 3 --------------------------------------------------------------

# Read in training and testing data - Y is binary, X1...Xn are cont. 
q3_train_data <- data.frame(read.csv("Midterm/Midterm-Data/Problem3training.csv"))
q3_test_data <- data.frame(read.csv("Midterm/Midterm-Data/Problem3testing.csv"))
q3_train_n <- nrow(q3_train_data)
q3_test_n <- nrow(q3_test_data)
q3_p <- ncol(q3_train_data) - 1

## i. Fit logistic, LDA, QDA, and SVM-Radial, and find their error rates

# Store errors
q3_errors <- c()

# Logistic model

q3_logit <- glm(Y ~ ., 
                data = q3_train_data,
                family = binomial)

q3_test_pred_logit <- 1*(predict(q3_logit, newdata = q3_test_data, type="response") > 0.5)

q3_errors[1] <- mean(q3_test_pred_logit != (q3_test_data$Y))

# LDA

q3_lda <- lda(Y ~ ., 
              data = q3_train_data)

q3_test_pred_lda <- as.numeric(predict(q3_lda, 
                                    newdata = q3_test_data)$class) - 1

q3_errors[2] <- mean(q3_test_pred_lda != (q3_test_data$Y))

# QDA

q3_qda <- qda(Y ~ ., 
              data = q3_train_data)

q3_test_pred_qda <- as.numeric(predict(q3_qda, 
                                    newdata = q3_test_data)$class) - 1
q3_errors[3] <- mean(q3_test_pred_qda != (q3_test_data$Y))

# SVM with a radial kernel

q3_svm_rad <- tune(svm,
                   as.factor(Y) ~ .,
                   data = q3_train_data,
                   kernel = "radial",
                   ranges=list(cost=c(0.01, 1, 5, 10, 100),
                               gamma=c(0.001, 0.01, 0.1, 1))) 
q3_svm_rad_fit <- q3_svm_rad$best.model

q3_test_pred_svm_rad <- (predict(q3_svm_rad_fit, newdata = q3_test_data))

q3_errors[4] <- mean(q3_test_pred_svm_rad != (q3_test_data$Y))

# Print error rates
q3_i_out <- cbind(q3_errors[1], q3_errors[2], q3_errors[3], q3_errors[4])
colnames(q3_i_out) <- c("Logistic", "LDA", "QDA", "SVM-R")
kable(q3_i_out, digits = 4, align = "l")

## ii. Plot the decision boundary for covariates X1 and X2 for all models

# Create grid of points for plots; X3...Xp will be set to zero
gridX1 <- seq(-3, 3, length = 100)
gridX2 <- seq(-3, 3, length = 100)
gridX <- expand.grid(gridX1, gridX2)
gridX_Zeros <- data.frame(matrix(0, 10000, q3_p-2))

q3_plot_data <- bind_cols(gridX, gridX_Zeros)

# Rename all columns to X1...Xp
colnames(q3_plot_data) <- c("X.1", "X.2", "X.3", "X.4", "X.5", "X.6", "X.7",
                            "X.8", "X.9", "X.10", "X.11", "X.12","X.13", "X.14",
                            "X.15", "X.16", "X.17", "X.18", "X.19", "X.20", "X.21",
                            "X.22", "X.23", "X.24", "X.25", "X.26", "X.27", "X.28") # Attempted a more efficient way, settled on brute force

# Run predictions and plot for each model

# Logistic model decision boundary
q3_logit_grid_pred <- 1*(predict(q3_logit, 
                                 q3_plot_data, 
                             type = "response") > 0.5)

q3_logit_plot <- q3_plot_data %>% 
  ggplot(aes(x = X.1, y = X.2, color = as.factor(q3_logit_grid_pred))) +
  geom_point() + 
  scale_colour_manual(values=c("blue", "darkgreen")) +
  labs(title = "Logsitic Regression", 
       x = "X1",
       y = "X2",
       color = "Outcome")

# LDA decision boundary
q3_lda_grid_pred <- as.numeric(predict(q3_lda, 
                                       newdata = q3_plot_data)$class) - 1

q3_lda_plot <- q3_plot_data %>% 
  ggplot(aes(x = X.1, y = X.2, color = as.factor(q3_lda_grid_pred))) +
  geom_point() + 
  scale_colour_manual(values=c("blue", "darkgreen")) +
  labs(title = "LDA", 
       x = "X1",
       y = "X2",
       color = "Outcome")

# QDA decision boundary
q3_qda_grid_pred <- as.numeric(predict(q3_qda, 
                                       newdata = q3_plot_data)$class) - 1

q3_qda_plot <- q3_plot_data %>% 
  ggplot(aes(x = X.1, y = X.2, color = as.factor(q3_qda_grid_pred))) +
  geom_point() + 
  scale_colour_manual(values=c("blue", "darkgreen")) +
  labs(title = "QDA", 
       x = "X1",
       y = "X2",
       color = "Outcome")

# SVM-Radial decision boundary
q3_svm_grid_pred <- (predict(q3_svm_rad_fit, newdata = q3_plot_data))

q3_svm_plot <- q3_plot_data %>% 
  ggplot(aes(x = X.1, y = X.2, color = as.factor(q3_svm_grid_pred))) +
  geom_point() + 
  scale_colour_manual(values=c("blue", "darkgreen")) +
  labs(title = "SVM-Radial", 
       x = "X1",
       y = "X2",
       color = "Outcome")

# Plot decision boundaries for each model on same grid
grid.arrange(q3_logit_plot, q3_lda_plot, q3_qda_plot, q3_svm_plot, ncol=2)

## iii. Use cross-validation to estimate the testing error rate for each approach

# 1. Find the testing error rate from cross-validation

# For cross-validation, we will use 10 folds to evaluate each approach
folds <- 10
groups <- cut(1:q3_train_n, breaks = 10, labels = F)

# Keep track of the error for each approach
q3_errors_CV <- matrix(NA, folds, 4)

for(fold in 1:folds){
  
  # Split the data into training/testing
  testIndex <- which(groups == fold)
  
  q3_train_CV <- q3_train_data[-testIndex,]
  q3_test_CV <- q3_train_data[testIndex,]
  
  # Logistic
  q3_logit_CV <- glm(Y ~ ., 
                  data = q3_train_CV,
                  family = binomial)
  
  q3_test_pred_logit_CV <- 1*(predict(q3_logit_CV, newdata = q3_test_CV, type="response") > 0.5)
  
  q3_errors_CV[fold, 1] <- mean(q3_test_pred_logit_CV != (q3_test_CV$Y))
  
  # LDA
  q3_lda_CV <- lda(Y ~ ., 
                   data = q3_train_CV)
  
  q3_test_pred_lda_CV <- as.numeric(predict(q3_lda_CV, 
                                         newdata = q3_test_CV)$class) - 1
  
  q3_errors_CV[fold, 2] <- mean(q3_test_pred_lda_CV != (q3_test_CV$Y))
  
  # QDA
  q3_qda_CV <- qda(Y ~ ., 
                   data = q3_train_CV)
  
  q3_test_pred_qda_CV <- as.numeric(predict(q3_qda_CV, 
                                            newdata = q3_test_CV)$class) - 1
  
  q3_errors_CV[fold, 3] <- mean(q3_test_pred_qda_CV != (q3_test_CV$Y))
  
  # SVM-Radial
  q3_svm_rad_CV <- tune(svm,
                     as.factor(Y) ~ .,
                     data = q3_train_CV,
                     kernel = "radial",
                     ranges=list(cost=c(0.01, 1, 5, 10, 100),
                                 gamma=c(0.001, 0.01, 0.1, 1))) 
  q3_svm_rad_fit_CV <- q3_svm_rad_CV$best.model
  
  q3_test_pred_svm_rad_CV <- (predict(q3_svm_rad_fit_CV, newdata = q3_test_CV))
  
  q3_errors_CV[fold,4] <- mean(q3_test_pred_svm_rad_CV != (q3_test_CV$Y))
}

# Find the avg error rate for each approach across the 10 folds
q3_logit_error_CV <- mean(q3_errors_CV[,1])
q3_lda_error_CV <- mean(q3_errors_CV[,2])
q3_qda_error_CV <- mean(q3_errors_CV[,3])
q3_svm_rad_error_CV <- mean(q3_errors_CV[,4])

# Print error rates
q3_iii_1_out <- cbind(q3_logit_error_CV, q3_lda_error_CV, q3_qda_error_CV, q3_svm_rad_error_CV)
colnames(q3_iii_1_out) <- c("Logistic", "LDA", "QDA", "SVM-R")
kable(q3_iii_1_out, digits = 4, align = "l")

# 2. Compare these error rates to those found in part (i)

# Print the difference in error rates: CV error rate - test error rate
q3_logit_diff <- q3_logit_error_CV - q3_errors[1]
q3_lda_diff <- q3_lda_error_CV - q3_errors[2]
q3_qda_diff <- q3_qda_error_CV - q3_errors[3]
q3_svm_rad_diff <- q3_svm_rad_error_CV - q3_errors[4]

q3_iii_2_out <- cbind(q3_logit_diff, q3_lda_diff, q3_qda_diff, q3_svm_rad_diff)
colnames(q3_iii_2_out) <- c("Logistic Diff.", "LDA Diff.", "QDA Diff.", "SVM-R Diff.")
kable(q3_iii_2_out, digits = 4, align = "l")

# 3. Give potential reasons for the cross-validation error rates and the error rates from part (i)

# Answer in markdown

## iv. Incorporate PCA within the SVM

# 1. Find the smallest M such that the principle components explain > 90% of the information from the original covariates

# Run PCA on the training data
q3_pca <- prcomp(q3_train_data, scale = TRUE)
summary(q3_pca)

# Compute cumulative explained variance
cumulative_var <- data.frame(x = c(1:29), y = as.numeric(cumsum(q3_pca$sdev^2 / sum(q3_pca$sdev^2)))) # Cumulative variance explained
colnames(cumulative_var) <- c("PC", "CumVar")

# Find smallest M such that cumulative variation > 90%
q3_prin_comp <- which(cumulative_var$CumVar >= 0.9)[1] # Takes the first index in which cumulative variation >= 0.9

# Plot PCA to visualize the proportion of variance explained by M components
cumulative_var %>%
  ggplot(aes(x = PC, y = CumVar)) +
  geom_point() +
  geom_hline(yintercept = 0.9, # Draw a horizontal line at 0.9, for 90% variation explained by the data set
             color = "blue", 
             linetype = "dashed") +
  geom_vline(xintercept = q3_prin_comp, # Draw a vertical line at M
             color = "red",
             linetype = "dashed") +
  xlab("PCs") +
  ylab("Cumulative Explained Variance")

# Use PCA with SVM-Radial
q3_pca_train_data <- q3_train_data[,1:(q3_prin_comp+1)] # Use first M principle components, add 1 to account for Y

q3_svm_rad_pca <- tune(svm,
                   as.factor(Y) ~ .,
                   data = q3_pca_train_data,
                   kernel = "radial",
                   ranges=list(cost=c(0.01, 1, 5, 10, 100),
                               gamma=c(0.001, 0.01, 0.1, 1))) 
q3_svm_rad_pca_fit <- q3_svm_rad_pca$best.model

# 2. Use those M principle components with an SVM and get the prediction error rate on the test data set
q3_test_pred_svm_rad_pca <- (predict(q3_svm_rad_pca_fit, newdata = q3_test_data))

q3_pca_error <- mean(q3_test_pred_svm_rad_pca != (q3_test_data$Y))

# Print out SVM error rate with M principle components
q3_iv_out <- cbind(q3_pca_error)
colnames(q3_iv_out) <- c("SVM-R with PCA")
kable(q3_iv_out, digits = 4, align = "l")


# Question 4 --------------------------------------------------------------

# Read in training and testing data - Y, X1...Xn are cont. 
q4_train_data <- data.frame(read.csv("Midterm/Midterm-Data/Problem4training.csv"))
q4_test_data <- data.frame(read.csv("Midterm/Midterm-Data/Problem4testing.csv"))
q4_train_n <- nrow(q4_train_data)
q4_test_n <- nrow(q4_test_data)
q4_p <- ncol(q4_train_data) - 1

## i. Make a heatmap of the empirical correlation matrix of the covariates (the (i,j) element is corr(Xi, Xj)) and interpret your findings

# Find correlation matrix
corr_mat <- round(cor(q4_train_data[,2:(q4_p+1)]), 2)
corr_map <- melt(corr_mat)

# Plot as a heatmap
corr_map %>%
  ggplot(aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", 
                      high = "red",
                      mid = "white",
                      midpoint = 0,
                      limit = c(-1,1)) +
  coord_fixed() +
  labs(title = "Empricial Correlation Matrix")

# Plot as a heatmap but justify the correlations to the lower triangle matrix
corr_mat %>% ggcorrplot(hc.order = TRUE,
                        type = "lower",
                        lab = FALSE) +
  labs(title = "Empricial Correlation Matrix - Rearranged") +
  xlab("Var1") +
  ylab("Var2")

## ii. Does the plot in part (i) tell you anything about whether it would be better to use dimension reduction (PCA, PLS) or shrinkage approaches (lasso, ridge)?

# Answer in markdown

## iii. Does the plot in part (i) tell you anything about whether it would be better to use a ridge or lasso penalty when using shrinkage approaches?

# Answer in markdown

## iv. Fit lasso regression, ridge regression, PLS, and PCR, and find their error rates

# Store errors
q4_errors <- c()

# Lasso regression
q4_lasso_cv <- cv.glmnet(x = as.matrix(q4_train_data[,2:(q4_p + 1)]), y = q4_train_data$Y, alpha = 1)
q4_lasso_param <- q4_lasso_cv$lambda.min # Choose lambda that minimizes test MSE using 10-fold CV

# Fit lasso regression with optimal lambda
q4_lasso <- glmnet(x = as.matrix(q4_train_data[,2:(q4_p + 1)]), y = q4_train_data$Y, alpha = 1, lambda = q4_lasso_param)
q4_test_pred_lasso <- predict(q4_lasso, s = q4_lasso_param, newx = as.matrix(q4_test_data[,2:(q4_p + 1)]))

q4_errors[1] <- mean((as.numeric(q4_test_pred_lasso) - q4_test_data$Y)^2) # Test MSE

# Ridge regression
q4_ridge_cv <- cv.glmnet(x = as.matrix(q4_train_data[,2:(q4_p + 1)]), y = q4_train_data$Y, alpha = 0)
q4_ridge_param <- q4_ridge_cv$lambda.min # Choose lambda that minimizes test MSE using 10-fold CV

# Fit ridge regression with optimal lambda
q4_ridge <- glmnet(x = as.matrix(q4_train_data[,2:(q4_p + 1)]), y = q4_train_data$Y, alpha = 0, lambda = q4_ridge_param)
q4_test_pred_ridge <- predict(q4_ridge, s = q4_ridge_param, newx = as.matrix(q4_test_data[,2:(q4_p + 1)]))

q4_errors[2] <- mean((as.numeric(q4_test_pred_ridge) - q4_test_data$Y)^2) # Test MSE

# PLS
q4_pls_cv <- plsr(Y ~ ., data = q4_train_data, scale = TRUE, validation = "CV")
q4_pls_components <- which.min(RMSEP(q4_pls_cv)$val[1,,])  - 1 # This will pull the number of components with the smallest CV error; subtract 1 to account for model w/ 0 PCs

# Fit PLS with M number of components found from cross-validation
q4_pls <- plsr(Y ~ ., data = q4_train_data, scale = TRUE, ncomp = q4_pls_components)
q4_test_pred_pls <- predict(q4_pls, q4_test_data, ncomp = q4_pls_components)

q4_errors[3] <- mean((as.numeric(q4_test_pred_pls) - q4_test_data$Y)^2) # Test MSE

# PCR
q4_pcr_cv <- pcr(Y ~ ., data = q4_train_data, scale = TRUE, validation = "CV")
q4_pcr_components <- which.min(RMSEP(q4_pcr_cv)$val[1,,]) - 1 # This will pull the number of components with the smallest CV error; subtract 1 to account for model w/ 0 PCs

# Fit PCR with M number of components found from cross-validation
q4_pcr <- pcr(Y ~ ., data = q4_train_data, scale = TRUE, ncomp = q4_pcr_components)
q4_test_pred_pcr <- predict(q4_pcr, q4_test_data, ncomp = q4_pcr_components)

q4_errors[4] <- mean((as.numeric(q4_test_pred_pcr) - q4_test_data$Y)^2) # Test MSE

# Print MSEs
q4_iv_out <- cbind(q4_errors[1], q4_errors[2], q4_errors[3], q4_errors[4])
colnames(q4_iv_out) <- c("Lasso", "Ridge", "PLS", "PCR")
kable(q4_iv_out, digits = 4, align = "l")

## v. Use two distinct variable selection techniques and discuss the pros/cons of each approach

# Answer in markdown

# Generate an intercept-only model and the full model
q4_base_model <- lm(Y ~ 1, data = q4_train_data)
q4_full_model <- lm(Y ~ ., data = q4_train_data)

# Forward stepwise regression
q4_fwd_reg <- step(q4_base_model, 
                   direction = "forward", 
                   scope = list(lower = q4_base_model, upper = q4_full_model), 
                   k = log(q4_train_n), 
                   trace = 0) # Trace = 0 suppresses display

q4_fwd_reg_coef <- length(q4_fwd_reg$coefficients) # Find number of covariates

# Both stepwise regression
q4_both_reg <- step(q4_base_model, 
                   direction = "both", 
                   scope = list(lower = q4_base_model, upper = q4_full_model), 
                   k = log(q4_train_n), 
                   trace = 0) # Trace = 0 suppresses display

q4_both_reg_coef <- length(q4_both_reg$coefficients) # Find number of covariates

# Print values
q4_v_out <- cbind(q4_fwd_reg_coef, q4_both_reg_coef)
colnames(q4_v_out) <- c("Forward - # of Covariates", "Both - # of Covariates")
kable(q4_v_out, digits = 4, align = "l")

# vi. Suppose you fit the ridge regression estimator that is given by ̂β.

# 1. Derive the variance of the ridge regression estimator.

# Answer in markdown
  
# 2. Set λ = 4.7 and σ2 = 1. Estimate the covariance between ̂β3 and ̂β55.
q4_I_p <- as.matrix(diag(q4_p)) # Identity matrix
q4_x_mat <- as.matrix(q4_train_data[,2:(q4_p + 1)])
q4_y_mat <- as.matrix(q4_train_data[,1])
q4_sigma_sq <- 1
q4_lambda <- 4.7
q4_var_cov_mat <- (t(q4_x_mat)%*%q4_x_mat + q4_lambda*q4_I_p)^-1 %*%
  (t(q4_x_mat)%*%q4_x_mat + q4_lambda*q4_I_p)^-1 %*%
  (t(q4_x_mat)%*%q4_x_mat) *
  q4_sigma_sq # This is the variance we found in part 1, which produces a variance-covariance matrix

# Cov(B3, B55) will thus be given by the value at [3,55]
q4_cov_B3_B55 <- q4_var_cov_mat[3,55]

# Print covariance estimate
q4_vi_out <- cbind(q4_cov_B3_B55)
colnames(q4_vi_out) <- c("Covariance Between B3 and B55")
kable(q4_vi_out, digits = 4, align = "l")
  
# 3. Is it possible to construct such an interval that is also valid? If so, implement it for covariate 20 
# and interpret your results. If not, explain why one can not be constructed.

q4_std_errors <- sqrt(abs(diag(q4_var_cov_mat))) # This will return the square root of the variances (the std. errors)

q4_estimates <- (t(q4_x_mat) %*% q4_x_mat + q4_lambda*q4_I_p)^-1 %*% 
  (t(q4_x_mat)%*%q4_y_mat) # Find the estimates of B

# Answer in markdown
