# Example 9.1 Heteroscedasticity-Consistent Variance-Covariance Matrix

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/greene.txt",header=T,nrows=51)
data<-na.omit(data)  # take care of missing obs #34

spending<-data$SPENDING
income<-data$INCOME/10000

results<-lm(spending~income+I(income^2))
summary(results)

# library("car")
# hccm(results)  # assuming heteroscedasticity (White-Huber estimator)

library("sandwich")
vcov(results)   # assuming homoscedasticity
vcovHC(results) # assuming heteroscedasticity
coeftest(results, vcov. = vcovHC)

rm(list=ls())
