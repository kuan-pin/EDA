# Example 7.3 Hypothesis Testing for Nonlinear Models

setwd("C:/Course17/ceR/R")
# Hypothesis Testing 
# Restricted CES Production Function: b[4]=1/b[3]
# Judge, et. al. [1988], Chapter 12

judge<- read.table("http://web.pdx.edu/~crkl/ceR/data/judge.txt")
# file name may be case sensitive!
summary(judge)

# log-likelihood function of unretricted model (5 parameters)
llf<-function(b) {
  l<-judge[,1]
  k<-judge[,2]
  q<-judge[,3]
  e<-log(q)-b[1]-b[4]*log(b[2]*l^b[3]+(1-b[2])*k^b[3])
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  ll<-ll+log(1/q)  # add log(jacobian)
  return(ll)
}
# log-likelihood function of retricted model (3 parameters)
# restrictions: b[4]=1/b[3]
llfr<-function(b) {
  l<-judge[,1]
  k<-judge[,2]
  q<-judge[,3]
  e<-log(q)-b[1]-(1/b[3])*log(b[2]*l^b[3]+(1-b[2])*k^b[3])
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  ll<-ll+log(1/q)  # add log(jacobian)
  return(ll)
}

# constraint function: b[4]=1/b[3]
cf<-function(b) {
  c(b[4]*b[3]-1)
}

library(maxLik)
library(numDeriv)

# Lagrangian Multiplier Test: based on constrained model
M1<-maxLik(llfr,start=c(1,0.5,-1))
summary(M1)
ll1<-M1$maximum  # maximum log-likelihood
beta1<-c(M1$estimate,1/M1$estimate[3])  # original parameterization
dll<-jacobian(llf,beta1) # jacobian (gradient matrix for vector-valued llf)
gll<-colSums(dll)  # gradient vector of llf
# use OPG to approximate var-cov(dll)
lmtest<-t(gll)%*%solve(t(dll)%*%dll)%*%gll

# Wald Test: based on unconstrained model
M2<-maxLik(llf,start=c(1,0.5,-1,-1))
summary(M2)
ll2<-M2$maximum
beta2<-M2$estimate
c2<-cf(beta2)
gc2<-jacobian(cf,beta2)
wtest<-t(c2)%*%solve(gc2%*%vcov(M2)%*%t(gc2))%*%c2

# Likelihood Ratio Test
lrtest<--2*(ll1-ll2)

test<-list(c("Wald Test",wtest),
           c("Lagrangian Multiplier Test",lmtest),
           c("Likelihood Ratio Test",lrtest))           
test
rm(list=ls())
