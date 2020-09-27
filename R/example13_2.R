# Example 13.2 Klein's Model I
# Simultaneous Equations Estimation
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
IV <- ~G+T+W2+A+K1+P1+X1

# define the system
system<-list(Consumption=eqC,Investment=eqI,Wages=eqW)

library(systemfit)
# OLS
klein.ols<-systemfit(system,data=Kdata)
summary(klein.ols)

# 2SLS
klein.2sls<-systemfit(system,"2SLS",inst=IV,data=Kdata,
                      methodResidCov="noDfCor")
summary( klein.2sls)

# 3SLS
klein.3sls<-systemfit(system,"3SLS",inst=IV,data=Kdata,
                      methodResidCov="noDfCor")
summary(klein.3sls)

# I3SLS
klein.I3sls<-systemfit(system,"3SLS",inst=IV,data=Kdata,
                       methodResidCov="noDfCor",maxit=100)
summary(klein.I3sls)

# alternative eqConsump with restriction
eqC1<-C~P+P1+W1+W2
system1<-list(Consumption=eqC1,Investment=eqI,Wages=eqW)
restrict1<-"Consumption_W1-Consumption_W2=0"
klein.I3slsr<-systemfit(system1,"3SLS",inst=IV,data=Kdata,restrict.matrix=c(restrict1),
                        methodResidCov="noDfCor",maxit=100)
summary(klein.I3slsr)

rm(list = ls())
