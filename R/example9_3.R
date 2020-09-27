# Example 9.3 Breusch-Pagan and White Tests

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/greene.txt",header=T,nrows=51)
data<-na.omit(data)  # take care of missing obs

spending<-data$SPENDING
income<-data$INCOME/10000
nobs<-length(spending)

results<-lm(spending~income+I(income^2))
e2<-results$resid^2
summary(results)

# using lmtest package 
library(lmtest)

# Breusch-Pagan Test for homoscedasticity
bptest(results)
bptest(results,studentize=F)

# White Test for homoscedasticity
results1<-lm(e2~income+I(income^2)+I(income^4)+I(income*income^2))
summary(results1)
NRstat<-nobs*summary(results1)$r.squared
NRdf<-results1$rank-1          # Chisq-test
NRstat
1-pchisq(NRstat,NRdf)

rm(list=ls())

