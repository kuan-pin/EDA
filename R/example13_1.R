# Example 13.1 Klein's Model I
# Single Equation Estimation
#
Kdata<-read.table("http://web.pdx.edu/~crkl/ceR/data/klein.txt",header=T,nrows=22)
# Year: 1920 -1941 
# C: Consumption in billions of 1934 dollars.
# P: Private profits.
# I: Investment.
# W1: Private wage bill.
# W2: Government wage bill.
# G: Government nonwage spending.
# T: Indirect taxes plus net exports.
# X: Total private income before taxes, or
# X = Y + T - W2 where Y is after taxes income.
# K1: Capital stock in the begining year, or
# capital stock lagged one year. 
# K1[1942]=209.4
Kdata$P1<-c(NA,Kdata$P[2:length(Kdata$P)-1])
Kdata$X1<-c(NA,Kdata$X[2:length(Kdata$X)-1])
Kdata$W<-Kdata$W1+Kdata$W2
Kdata$K<-c(Kdata$K1[2:length(Kdata$K1)],209.4)
Kdata$A<-Kdata$YEAR-1931
Kdata<-subset(Kdata,Kdata$YEAR>1920)
summary(Kdata)

# define equation formula
eqC<-C~P+P1+W
eqI<-I~P+P1+K1
eqW<-W1~X+X1+A
  
# define instruments
IV<- ~ G+T+W2+A+K1+P1+X1

# OLS
C.ols<-lm(eqC,data=Kdata)
summary(C.ols)
I.ols<-lm(eqI,data=Kdata)
summary(I.ols)
W.ols<-lm(eqW,data=Kdata)
summary(W.ols)

# 2SLS or IV
library(AER)
C.iv<-ivreg(eqC,IV,data=Kdata)
summary(C.iv)
I.iv<-ivreg(eqI,IV,data=Kdata)
summary(I.iv)
W.iv<-ivreg(eqW,IV,data=Kdata)
summary(W.iv)

rm(list = ls())
