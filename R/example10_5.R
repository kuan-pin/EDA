# Example 10.5 Higher Order Autocorrelation

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year <- data$YEAR
X <- log(data$X)
L <- log(data$L1)
K <- log(data$K1)
results.ols<-lm(X~L+K)
summary(results.ols)

# cochrane-orcutt lm, given AR(p) rho, derived from ols model
lm.ar <- function(model,rho) {
  x <- model.matrix(model)
  y <- model.response(model.frame(model))
  r <- c(1,-rho)
  ystar<-filter(y,r,sides=1)
  xstar<-filter(x,r,sides=1)
  return(lm(ystar~xstar-1))
}

# cochrane-orcutt procedure
lm.co<-function(model,order) {
  x<-model.matrix(model)
  y<-model.response(model.frame(model))
  e<-y-x%*%model$coefficients
  p<-order+1
  estar<-embed(e,p)
  r0<-lm(estar[,1]~estar[,2:p]-1)$coefficients
  rdiff<-1
  while(rdiff>1.0e-5) {
    model1<-lm.ar(model,r0)
    e<-y-x%*%model1$coefficients
    estar<-embed(e,p)
    rho<-lm(estar[,1]~estar[,2:p]-1)$coefficients
    rdiff<-sum((rho-r0)^2)
    r0<-rho
  }
  list(lm.co=summary(lm.ar(model,r0)),rho=r0)
}
results.co<-lm.co(results.ols,3) # the results are based on transformed model
results.co

# check for residual autocorrelation
e<-results.co[[1]]$residuals
for(i in 1:4) print(Box.test(e,lag=i,type="Box-Pierce"))
for(i in 1:4) print(Box.test(e,lag=i,type="Ljung-Box"))

acf(e,type="correlation")
acf(e,type="partial")  # same pacf(results$resid)

# alternatively using package nlme
library(nlme)
results.gls<-gls(X~L+K,correlation=corARMA(p=3), method="ML")
summary(results.gls)

rm(list=ls())
