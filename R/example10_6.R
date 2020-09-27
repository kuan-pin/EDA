# Example 10.6 ARMA(1,1) Error Structure

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year <- data$YEAR
X <- log(data$X)
L <- log(data$L1)
K <- log(data$K1)

results.arma<-arima(X,order=c(1,0,1),xreg=cbind(L,K),method="ML")
results.arma  
  
# check for residual autocorrelation
for(i in 1:4) print(Box.test(results.arma$resid,lag=i,type="Box-Pierce"))
for(i in 1:4) print(Box.test(results.arma$resid,lag=i,type="Ljung-Box"))

acf(results.arma$resid,type="correlation")
acf(results.arma$resid,type="partial")  # same pacf(results$resid)

# alternatively using package nlme
library(nlme)
results.gls<-gls(X~L+K,correlation=corARMA(p=1,q=1), method="ML")
summary(results.gls)

rm(list=ls())
