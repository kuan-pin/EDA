# Example 14.2 Cointegration Test
# Engle-Granger Approach
# Using package urca, egcm

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",header=T,nrows=66)
Y<-ts(data[,2],start=1929)
C<-ts(data[,3],start=1929)

co1<-lm(C~Y)
summary(co1)
X<-co1$residuals

# apply unit-roots test on residuals with Model I
# need to use eg-test critical values
library(urca)
# model I
test.df1<-ur.df(X,"none",selectlags='AIC')
summary(test.df1)

# alternatively, 
# using ca.po(): Phillips-Ouliaris Cointegration test
# using egcm: install.packages("egcm")
library(egcm)
eg1<-egcm(C,Y, log=FALSE,include.const=TRUE)
summary(eg1)

rm(list=ls())
