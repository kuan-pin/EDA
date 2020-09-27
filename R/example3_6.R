# Example 3.6
# Residual Diagnostics

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year<-data$YEAR
X<-log(data$X)
L<-log(data$L1) 
K<-log(data$K1)
nobs<-length(X)

results<-lm(X ~ L+K)
summary(results)
# Bera-Jarque Test for asymptotic normality
e<-results$resid   # residuals
v<-mean(e^2)       # asymp. variance
s<-mean((e/sqrt(v))^3)  # skewness
k<-mean((e/sqrt(v))^4)  # kurtosis
JBstat<-nobs*((s^2)/6+((k-3)^2)/24)
JBstat
1-pchisq(JBstat,2)

# check for residual normality
qqnorm(results$residuals)
qqline(results$residuals)

# Using tseries package
# library(tseries)
# jarque.bera.test(results$resid)

Standard_res<-rstandard(results)
Student_res<-rstudent(results)
Leverage<-hatvalues(results)
DFFITS<-dffits(results)

cbind(Leverage,Standard_res,Student_res,DFFITS)

rm(list=ls())
