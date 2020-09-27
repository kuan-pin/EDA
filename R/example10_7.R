# Example 10.7 Maximum Likelihood Estimation
# AR(1), MA(1), ARMA(1,1)
# Using package maxLik

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year <- data$YEAR
X <- log(data$X)
L <- log(data$L1)
K <- log(data$K1)
results.ols<-lm(X~L+K)
summary(results.ols)

# define log-likelihood function of AR(1) error structure
llf.ar1<-function(b) {
  y<-X
  x<-cbind(1,L,K)
  k<-ncol(x)
  e<-y-x%*%b[1:k]
  n<-nrow(e)
  u<-e-b[k+1]*c(NA,e[1:(n-1)]) # AR(1)=b[k+1]
  u[1]<-sqrt(1-b[k+1]^2)*e[1]
  # u[2:n]<-e[2:n]-b[k+1]*e[1:(n-1)]
  llf<-dnorm(u,mean=0,sd=sqrt(mean(u^2)),log=T)
  # jacobian=c(sqrt(1-b[k+1]^2),rep(1,n-1))
  llj<-log(c(sqrt(1-b[k+1]^2),rep(1,n-1))) 
  as.vector(llf+llj)  # log-jacobian included
}

# define log-likelihood function of MA(1) error structure
llf.ma1<-function(b) {
  y<-X
  x<-cbind(1,L,K)
  k<-ncol(x)
  e<-y-x%*%b[1:k]
  n<-nrow(e)
  # MA(1)=b[k+1] series with recursive filter
  u<-filter(e,b[k+1],method="recursive")
  as.vector(dnorm(u,mean=0,sd=sqrt(mean(u^2)),log=T))
}

# define log-likelihood function of ARMA(1,1) error structure
llf.arma1<-function(b) {
  y<-X
  x<-cbind(1,L,K)
  k<-ncol(x)
  e<-y-x%*%b[1:k]
  n<-nrow(e)
  u<-e-b[k+1]*c(NA,e[1:(n-1)]) # AR(1)=b[k+1], MA(1)=b[k+2]
  u[1]<-sqrt(1-b[k+1]^2)*e[1]
  # u[2:n]<-e[2:n]-b[k+1]*e[1:(n-1)]
  v<-filter(u,b[k+2],method="recursive")
  llf<-dnorm(v,mean=0,sd=sqrt(mean(v^2)),log=T)
  # jacobian=c(sqrt(1-b[k+1]^2),rep(1,n-1))
  llj<-log(c(sqrt(1-b[k+1]^2),rep(1,n-1))) 
  as.vector(llf+llj)  # log-jacobian included
}

library(maxLik)
results.ar1<-maxLik(llf.ar1,start=c(results.ols$coefficients,0),method="BHHH")
summary(results.ar1)
results.ma1<-maxLik(llf.ma1,start=c(results.ols$coefficients,0),method="BHHH")
summary(results.ma1)
results.arma1<-maxLik(llf.arma1,start=c(results.ols$coefficients,0,0),method="BHHH")
summary(results.arma1)

# alternatively using package nlme
library(nlme)
results1 <- gls(X~L+K,correlation=corARMA(p=1),method="ML")
summary(results1)
results2 <- gls(X~L+K,correlation=corARMA(q=1),method="ML")
summary(results2)
results3 <- gls(X~L+K,correlation=corARMA(p=1,q=1),method="ML")
summary(results3)

rm(list=ls())
