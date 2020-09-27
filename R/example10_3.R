# Example 10.3 Cochrane-Orcutt Iterative Procedure

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year <- data$YEAR
X <- log(data$X)
L <- log(data$L1)
K <- log(data$K1)
results.ols<-lm(X~L+K)
summary(results.ols)

# cochrane-orcutt lm, given AR(1) rho, derived from ols model
lm.ar1.co <- function(model,rho) {
  x <- model.matrix(model)
  y <- model.response(model.frame(model))
  n <- length(y)
  t <- 2:n
  ystar <- y[t] - rho * y[t-1]
  xstar <- x[t,] - rho * x[t-1,]
  return(lm(ystar ~ xstar - 1))
}

# prais-winsten lm given AR(1) rho, derived from ols model
lm.ar1.pw <- function(model,rho) {
  x <- model.matrix(model)
  y <- model.response(model.frame(model))
  n <- length(y)
  t <- 2:n
  ystar <- y[t] - rho * y[t-1]
  xstar <- x[t,] - rho * x[t-1,]
  ystar <- c(sqrt(1-rho^2)*y[1],ystar)
  xstar <- rbind(sqrt(1-rho^2)*x[1,],xstar)
  return(lm(ystar ~ xstar - 1))
}

# cochrane-orcutt procedure
lm.co<-function(model) {
  x<-model.matrix(model)
  y<-model.response(model.frame(model))
  e<-y-x%*%model$coefficients
  e1<-c(NA,e[1:length(e)-1])
  r0<-lm(e~e1-1)$coefficients
  rdiff<-1
  while(rdiff>1.0e-5) {
    model1<-lm.ar1.co(model,r0)
    e<-y-x%*%model1$coefficients
    e1<-c(NA,e[1:length(e)-1])
    rho<-lm(e~e1-1)$coefficients
    rdiff<-sum((rho-r0)^2)
    r0<-rho
  }
  list(lm.co=summary(lm.ar1.co(model,r0)),rho=r0)
}
results.co<-lm.co(results.ols) # the results are based on transformed model
results.co

# prais-winsten procedure
lm.pw<-function(model) {
  x<-model.matrix(model)
  y<-model.response(model.frame(model))
  e<-y-x%*%model$coefficients
  e1<-c(NA,e[1:length(e)-1])
  r0<-lm(e~e1-1)$coefficients
  rdiff<-1
  while(rdiff>1.0e-5) {
    model1<-lm.ar1.pw(model,r0)
    e<-y-x%*%model1$coefficients
    e1<-c(NA,e[1:length(e)-1])
    rho<-lm(e~e1-1)$coefficients
    rdiff<-sum((rho-r0)^2)
    r0<-rho
  }
  list(lm.pw=summary(lm.ar1.pw(model,r0)),rho=r0)
}
results.pw<-lm.pw(results.ols) # the results are based on transformed model
results.pw

# alternatively, using package orcutt, prais
library(orcutt)
results1 <- cochrane.orcutt(results.ols)
results1

library(prais)
results2<-prais.winsten(results.ols,data=NULL)
results2

library(nlme)
results3<-gls(X~L+K,corr=corAR1(),method="ML")
results3

rm(list=ls())
