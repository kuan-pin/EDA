# Example 7.2 Box-Cox Transformation
# U. S. Money Demand Equation
# Greene: Chapter 7

setwd("C:/Course17/ceR/R")
money<-read.table("http://web.pdx.edu/~crkl/ceR/data/money.txt",header=T)
# file name may be case sensitive!
summary(money)

m0<-money[,3]/1000  # money
r0<-money[,2]       # interest rate
y0<-money[,4]/1000  # income
model0<-lm(m0~r0+y0)
b0<-model0$coefficients

# log-likelihood function
llf<-function(b) {
  m<-(m0^b[5]-1)/b[5]   # money
  r<-(r0^b[4]-1)/b[4]   # interest rate
  y<-(y0^b[4]-1)/b[4]   # income
  e<-m-b[1]-b[2]*r-b[3]*y
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  lj<-(b[5]-1)*log(m0)  # log jacobian
  sum(ll+lj)
}
# model<-optim(c(b0,1,1),llf,method="BFGS",hessian=T,control=list(fnscale=-1))

llf5<-function(b) {
  m<-(m0^b[5]-1)/b[5]   # money
  r<-(r0^b[4]-1)/b[4]   # interest rate
  y<-(y0^b[4]-1)/b[4]   # income
  e<-m-b[1]-b[2]*r-b[3]*y
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  lj<-(b[5]-1)*log(m0)  # log jacobian
  ll+lj

}
#install.packages("maxLik")
# Maximum Likelihood
library(maxLik)   # need to install maxLik package
money5<-maxLik(llf5,start=c(b0,1,1),method="BHHH",control=list(iterlim=200))
summary(money5)

rm(list=ls())

