# Example 9.4 Multiplicative Heteroscedasticity
# Multiplicative Heteroscedasticity

setwd("C:/Course17/ceR/R")
greene<-read.table("http://web.pdx.edu/~crkl/ceR/data/greene.txt",header=T,nrows=51)
# file name may be case sensitive!

greene<-na.omit(greene)  # take care of missing obs
summary(greene)

x<-greene[,3]/10000  # income
y<-greene[,2]        # spending

model0<-lm(y~x+I(x^2))
b0<-model0$coefficients

# log-likelihood function (data: y,x)
llf1<-function(b) {
  h<-x^b[4]           # multiplicative hetero.
  # h<-exp(x*b[4])
  e<-(y-b[1]-b[2]*x-b[3]*(x^2))/sqrt(h)
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  ll-0.5*log(h)
}

llf2<-function(b) {
  # h<-x^b[4]           # multiplicative hetero.
  h<-exp(x*b[4])
  e<-(y-b[1]-b[2]*x-b[3]*(x^2))/sqrt(h)
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  ll-0.5*log(h)
}

# Maximum Likelihood
library(maxLik)   # need to install maxLik package
model1<-maxLik(llf1,start=c(b0,0),method="BHHH")
summary(model1)
model2<-maxLik(llf2,start=c(b0,0),method="BHHH")
summary(model2)

rm(list=ls())
