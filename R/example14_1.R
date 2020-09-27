# Example 14.1 Unit Root Tests
# Using package urca, tseries
setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",header=T,nrows=66)

Y<-ts(data[,2],start=1929)
C<-ts(data[,3],start=1929)

library(urca)  # using ur.df, ur.pp, ur.kpss, ... with urca package
# dickey-fuller unit-root test
# (1) which model? III, II, I
# (2) how many lags augmented?
X<-C  # select a variable for testing
# model III  
test.df3<-ur.df(X,"trend",selectlags='AIC')
summary(test.df3)
# model II 
test.df2<-ur.df(X,"drift",selectlags='AIC')
summary(test.df2)
# model I
test.df1<-ur.df(X,"none",selectlags='AIC')
summary(test.df1)

# alternatively, 
library(tseries)
adf.test(X)

rm(list=ls())
