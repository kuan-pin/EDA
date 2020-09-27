# Example 10.2 Tests for Autocorrelation
# Using package sandwich, lmtest
setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year <- data$YEAR
X <- log(data$X)
L <- log(data$L1)
K <- log(data$K1)
results<-lm(X~L+K)
summary(results)

library(lmtest)
# Durbin-Watson test
dwtest(results)
# Breusch-Godfrey test
bgtest(results)
for(i in 1:4) print(bgtest(results,order=i))

# check for residual autocorrelation
Box.test(results$resid,lag=12,type="Box-Pierce")
Box.test(results$resid,lag=12,type="Ljung-Box")

acf(results$resid,lag.max = 12)$acf
acf(results$resid,type="correlation")
pacf(results$resid,lag.max = 12)$acf
acf(results$resid,type="partial")  # same pacf(results$resid)

rm(list=ls())

