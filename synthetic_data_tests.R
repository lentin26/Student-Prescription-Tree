# Susan Athey Causal Forests for CATE E[Y2 - Y1|X=x]: https://grf-labs.github.io/grf/
# Kallus Recursive Partitioning for Personalization using Observational Data: https://github.com/Aida-Rahmattalabi/PersonalizationTrees
# Nuti Explainable Bayesian Decision Tree Algorithm: https://github.com/UBS-IB/bayesian_tree
# Dunnhumby Grocery store Data: https://www.dunnhumby.com/source-files/
# Estimating Treatment Effects with Causal Forests: https://arxiv.org/pdf/1902.07409.pdf

install.packages("grf")           #generalized random forests
install.packages("tidyverse")
install.packages("readxl")

rm(list = ls())

library(grf)
library(tidyr)
library(tidyverse)
library(dplyr)
library(grf)
library(binr)
library(timereg)
library(policytree)
library(DiagrammeR)

# Synthetic dataset 1 - linear probit model with no confounding
set.seed(39)
n <- 10000 # samples
h <- -1
X0 <- rnorm(n, 5, 1)
X1 <- rnorm(n, 5, 1)
g <- X0
P <- rnorm(n, 5, 1)
e <- rnorm(n, 0, 1)

# descrete price
t = as.numeric(quantile(P, probs=seq(0.1, 0.9, length.out=9)))
P.descrete <- sample(
  t, 
  size = n, 
  replace = TRUE,
  pnorm(seq(0.1, 0.9, length.out=9), 5, 1)
)

# binary indicator variable for purchase
Y <- as.numeric(g + h*P.descrete + e > 0) 

# visualize
hist(Y)
data <- generate.dataset.1()
data %>% ggplot(aes(x=X0, y=X1, color=P.descrete)) + geom_point()

# optimal revenue is sum of minimum price at which each customer is more likely to purchase than not
# i.e. Y.star = 0, we don't count negative prices, since this would require us to pay the customer to
# consume the product
P.optimal <- case_when(-g/h < min(P.descrete) ~ 0, TRUE ~ -g/h)
P.optimal <- case_when(
  -h/g < t[1] ~ 0, 
  t[1] <= -h/g & -h/g < t[2] ~ t[1],
  t[2] <= -h/g & -h/g < t[3] ~ t[2],
  t[3] <= -h/g & -h/g < t[4] ~ t[3],
  t[4] <= -h/g & -h/g < t[5] ~ t[4],
  t[5] <= -h/g & -h/g < t[6] ~ t[5],
  t[6] <= -h/g & -h/g < t[7] ~ t[6],
  t[7] <= -h/g & -h/g < t[8] ~ t[7],
  t[8] <= -h/g & -h/g < t[9] ~ t[8],
  t[9] <= -h/g ~ t[9]
  )

R.optimal <- sum(P.optimal)/length(P.optimal)

# 1vA binary treatment effects with causal forest
X <- data.frame(X0, X1)
W <- as.numeric(P.descrete == t[1])
R <- Y*P.descrete
tau.forest <- causal_forest(X, R, W)

# Estimate treatment effects for the training data using out-of-bag prediction.
tau.hat.oob <- predict(tau.forest)
hist(tau.hat.oob$predictions)

# maybe do this instead on 1vA?
multi.forest <- grf::multi_arm_causal_forest(X, R, as.factor(P.descrete))

# assign best treatment
# Compute doubly robust scores.
dr.scores <- double_robust_scores(multi.forest)
head(dr.scores)

# Fit a depth-2 tree on the doubly robust scores.
tree <- policy_tree(X, dr.scores, depth = 2)
plot(tree)

# assign treatments
predict <- predict(tree, X)
P.CT <- tree$action.names[predict]
Y.CT <- as.numeric(g + h*P.CT + e > 0) 

# revenue
R.CT <- sum(Y.CT*P.CT)/length(Y.CT)
print(c(R.CT, R.optimal))

################# Causal Trees Revenue ################# 
Predict.CT.Revenue <- function(data){
  R <- unlist(data %>% select(R), use.names=FALSE)
  P <- unlist(data %>% select(P), use.names=FALSE)
  h <- unlist(data %>% select(h), use.names=FALSE)
  g <- unlist(data %>% select(g), use.names=FALSE)
  X <- data %>% select(-c(R, P))
  
  multi.forest <- grf::multi_arm_causal_forest(X, R, as.factor(P))
  
  # assign best treatment
  # Compute doubly robust scores.
  dr.scores <- double_robust_scores(multi.forest)
  
  # Fit a depth-2 tree on the doubly robust scores.
  tree <- policy_tree(X, dr.scores, depth = 2)
  
  # assign treatments
  predict <- predict(tree, X)
  P.CT <- tree$action.names[predict]
  Y.CT <- as.numeric(h + g*P.CT + e > 0) 
  
  # revenue
  R.CT <- sum(Y.CT*P.CT)/length(Y.CT)
  
  # optimal revenue based on underlying probability distribution
  P.optimal <- case_when(-g/h < 0 ~ 0, TRUE ~ -g/h)
  R.optimal <- sum(P.optimal)/length(P.optimal)
  
  return(list(R.optimal, R.CT))
}
################# Causal Trees Revenue ################# 
generate.dataset.1 <- function(n){
  h <- -1
  X0 <- rnorm(n, 5, 1)
  X1 <- rnorm(n, 5, 1)
  g <- X0
  P <- rnorm(n, 5, 1)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  t = as.numeric(quantile(P, probs=seq(0.1, 0.9, length.out=9)))
  P.descrete <- sample(
    t, 
    size = n, 
    replace = TRUE,
    pnorm(seq(0.1, 0.9, length.out=9))
  )
  
  # binary indicator variable for purchase
  Y <- as.numeric(g + h*P.descrete + e > 0) 
  
  # 1vA binary treatment effects with causal forest
  X <- data.frame(X0, X1)
  R <- Y*P.descrete
  #Y <- as.factor(Y)
  P <- P.descrete
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X, P, R, Y)) 
}

data <- generate.dataset.1(10000)
Predict.CT.Revenue(data)

# sample from generative dataset 10 times and compute optimal and CT revenue
results <- data.frame()
for (i in 1:10){
  data <- generate.dataset.1()
  results <- rbind(results, Predict.CT.Revenue(data))
}

names(results) = c('Optimal', 'CT')

# Dataset 2 - higher dimension probit model with sparse linear interaction
generate.dataset.2 <- function(n){
  p <- 20
  X <- matrix(rnorm(n*p), n, p)
  g <- 5
  beta <- c(rnorm(5), rep(0, 15))
  
  h <- X %*% beta
  P <- rnorm(n, 0, 2)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  t = as.numeric(quantile(P, probs=seq(0.1, 0.9, length.out=9)))
  P <- sample(
    t, 
    size = n, 
    replace = TRUE,
    pnorm(seq(0.1, 0.9, length.out=9), 0, 2)
  )
  
  # binary indicator variable for purchase
  Y.star <- h + g*P.descrete + e
  Y <- as.numeric(Y.star > 0) 
  
  # data 
  X <- data.frame(X)
  R <- Y*P.descrete
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X, round(P,2), R, g, h)) 
}

#################### visualize ####################
# dataset 1
data <- generate.dataset.1(10000)

# overall, customers more likely to not buy
hist(data$Y)
mean(Y)

# demand curve
data %>% group_by(P) %>% summarise(Mean = mean(Y, na.rm=TRUE)) %>%
  ggplot(aes(x=P, y=Mean)) + geom_line()

# customers less likely to buy if prices are higher 
data %>% ggplot(aes(x=as.factor(Y), y=P)) + geom_boxplot(color="red", fill="orange", alpha=0.2)

# average revenue vs price plot
data %>% group_by(P) %>% summarise(Mean = mean(R, na.rm=TRUE)) %>%
  ggplot(aes(x=P, y=Mean)) + geom_line()

# price assignment - no confounding
data %>% ggplot(aes(x=X0, y=X1, color=P)) + geom_point()

##################### results ##################### 

library(data.table)
get.results <- function(n, dataset.generator){
  # function to find min, max and mean
  multi.fun <- function(x) {
    c(min = min(x), mean = mean(x), max = max(x))
  }
  
  # generate and test multiple samples
  result = list()
  for (i in 1:10){
    data <- dataset.generator(n)
    result <- rbindlist(list(result, Predict.CT.Revenue(data)))
  }
  
  # summarize and result results
  colnames(result) <- c('Optimal', 'CT')
  return(sapply(result, multi.fun))
}

get.results(100, generate.dataset.2)


############## delete ############## 
result = list()
for (i in 1:10){
  data <- generate.dataset.2(100)
  result <- rbindlist(list(result, Predict.CT.Revenue(data)))
  
}
colnames(result) <- c('Optimal', 'CT')
sapply(result, multi.fun)

