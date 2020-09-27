# Example 12.1 GMM Estimation of a Gamma Distribution
# Generalized Method of Moments
# Estimating Gamma Distribution of Income
# Using package gmm, numDeriv
#
setwd("C:/Course17/ceR/R")
yed<- read.table("http://web.pdx.edu/~crkl/ceR/data/yed20.txt",header=T,nrows=20)
# file name may be case sensitive!
summary(yed)
y<-yed$Income/10  # scale the data may help, y ~ prob. dist.

mf<-function(b,y) {
  m1<-y-b[1]/b[2]
  m2<-y^2-b[1]*(b[1]+1)/(b[2]^2)
  m3<-log(y)-digamma(b[1])+log(b[2]) # digamma() = deriv of log-gamma()
  m4<-1/y-b[2]/(b[1]-1)
  cbind(m1,m2,m3,m4)
}

library(gmm)
# using optimal weights matrix with HAC var-cov
gmm2<-gmm(mf,y,c(2,1),kernel="Truncated",vcov="iid")
summary(gmm2)
gmm3<-gmm(mf,y,c(2,1),kernel="Truncated",vcov="iid",type="iterative")
summary(gmm3)
gmm4<-gmm(mf,y,c(2,1),kernel="Truncated",vcov="iid",type="cue")
summary(gmm4)

# functions gmmm, gmmv, gmmf require user defined function mf
# sample moments function
gmmm<-function(b,data) {
  mf1<-mf(b,data)
  m<-colMeans(mf1)
  return(m)
}

# var-cov matrix of sample moments function, set p=0 for iid case
gmmv<-function(b,data,p) {
  mf1<-mf(b,data)
  m1<-mf1/nrow(mf1)
  v<-t(m1)%*%m1
  if(p>0) {
    i<-1
    while(i<=p) {
      s<-t(m1)%*%rbind(matrix(0,i,ncol(m1)),m1[(i+1):nrow(m1),])
      v<-v+(1-i/(p+1))*(s+t(s))
      i=i+1
    }
  }
  return(v)
}

# objective function to be minimized
gmmf<-function(b,data,wmatrix) {
  mf1<-mf(b,data)
  m<-colMeans(mf1)
  t(m)%*%wmatrix%*%m
}

# use optim to show GMM estimation (p=0 for iid case, data=y)
p<-0   # HAC with p-order serial correlation
gmm0<-optim(c(2,1),gmmf,data=y,wmatrix=diag(4),method="BFGS",hessian=T)
gmm0$value
gmm0$par

b0<-gmm0$par
V<-gmmv(b0,y,p)
W<-solve(V)
gmm1<-optim(b0,gmmf,data=y,wmatrix=W,method="BFGS",hessian=T)
gmm1$value
gmm1$par

iter<-1
diff<-sqrt(sum((b0-gmm1$par)^2))
while ((diff>1.0e-5) & (iter<100)) {
  b0<-gmm1$par
  W<-solve(gmmv(b0,y,p))
  gmm1<-optim(b0,gmmf,data=y,wmatrix=W,method="BFGS",hessian=T)
  diff<-sqrt(sum((b0-gmm1$par)^2))
  iter<-iter+1
}
iter
gmm1$value
gmm1$par

# compute var-cov matrix of b
library(numDeriv)
b1<-gmm1$par
W<-solve(gmmv(b1,y,p))
G<-jacobian(gmmm,b1,data=y)
vb1<-solve(t(G)%*%W%*%G)
b1
sqrt(diag(vb1))

rm(list = ls())

