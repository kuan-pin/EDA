# Example 10.4 Hildreth-Lu Grid Search Procedure

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

# Hildreth-Lu procedure based Cochrane-Orcutt transformation
lm.hl.co<-function(model) {
  r0<-0; s0<-1
  while(s0>1.0e-5) {
    rho<-seq(r0-0.9*s0,r0+0.9*s0,by=s0/10)
    sse<-sapply(rho, function(r) {deviance(lm.ar1.co(model,r))})
    tab<-data.frame("rho" = rho, "SSE" = sse)
    rho.min<-rho[which.min(tab$SSE)]
    r0<-rho.min
    s0<-s0/10
  }
  list(lm.hl=summary(lm.ar1.co(model,r0)),rho=r0)
}
results.hl.co<-lm.hl.co(results.ols) # the results are based on transformed model
results.hl.co

# Hildreth-Lu procedure based prais-winsten transformation
lm.hl.pw<-function(model) {
  r0<-0; s0<-1
  while(s0>1.0e-5) {
    rho<-seq(r0-0.9*s0,r0+0.9*s0,by=s0/10)
    sse<-sapply(rho, function(r) {deviance(lm.ar1.pw(model,r))})
    tab<-data.frame("rho" = rho, "SSE" = sse)
    rho.min<-rho[which.min(tab$SSE)]
    r0<-rho.min
    s0<-s0/10
  }
  list(lm.hl=summary(lm.ar1.pw(model,r0)),rho=r0)
}
results.hl.pw<-lm.hl.pw(results.ols) # the results are based on transformed model
results.hl.pw

rm(list=ls())

