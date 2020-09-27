# Example 10.1 Heteroscedasticity Autocorrelation
# Consistent Variance-Covariance Matrix
# Using library sandwich, lmtest
setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year <- data$YEAR
X <- log(data$X)
L <- log(data$L1)
K <- log(data$K1)
results<-lm(X~L+K)
summary(results)

library(sandwich)
vcov(results)  
vcovHAC(results)
NeweyWest(results)

library(lmtest)
coeftest(results, vcov. = vcovHAC)
coeftest(results, vcov. = NeweyWest)

rm(list=ls())
