# Example 12.2 A Nonlinear Rational Expectation Model
# Generalized Method of Moments
# A Nonlinear Rational Expectation Model
# GMM Estimation of Hansen-Singleton Model (Ea, 1982)
#
setwd("C:/Course17/ceR/R")
x<-read.table("http://web.pdx.edu/~crkl/ceR/data/gmmq.txt")
# data columns: (V1) c(t+1)/c(t) (V2)vwr (V3)rfr
summary(x)

# User-defined moments equations, named mf
mf<-function(b,x) {
  x0<-as.matrix(x[1:nrow(x)-1,])
  x1<-as.matrix(x[2:nrow(x),])   # lag of x
  z<-cbind(1,x1)                 # IV
  m1<-z*(b[1]*(x0[,1]^(b[2]-1)*x0[,2])-1)
  m2<-z*(b[1]*(x0[,1]^(b[2]-1)*x0[,3])-1)
  cbind(m1,m2)
}

library(gmm)
model1<-gmm(mf,x,c(1,0))  # default vcov="HAC"
summary(model1)
model2<-gmm(mf,x,c(1,0),type="iterative")
summary(model2)
model3<-gmm(mf,x,c(1,0),type="cue")
summary(model3)

model1a<-gmm(mf,x,c(1,0),kernel="Truncated",vcov="iid")
summary(model1a)
model2a<-gmm(mf,x,c(1,0),kernel="Truncated",vcov="iid",type="iterative")
summary(model2a)
model3a<-gmm(mf,x,c(1,0),kernel="Truncated",vcov="iid",type="cue")
summary(model3a)

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

# use optim to show GMM estimation (p=0 for iid case, data=x)
p<-0   # HAC with p-order serial correlation
gmm0<-optim(c(1,0),gmmf,data=x,wmatrix=diag(8),method="BFGS",hessian=T)
gmm0$value
gmm0$par

b0<-gmm0$par
V<-gmmv(b0,x,p)
W<-solve(V)
gmm1<-optim(b0,gmmf,data=x,wmatrix=W,method="BFGS",hessian=T)
gmm1$value
gmm1$par

iter<-1
diff<-sqrt(sum((b0-gmm1$par)^2))
while ((diff>1.0e-5) & (iter<100)) {
  b0<-gmm1$par
  W<-solve(gmmv(b0,x,p))
  gmm1<-optim(b0,gmmf,data=x,wmatrix=W,method="BFGS",hessian=T)
  diff<-sqrt(sum((b0-gmm1$par)^2))
  iter<-iter+1
}
iter
gmm1$value
gmm1$par

# compute var-cov matrix of b
library(numDeriv)
b1<-gmm1$par
W<-solve(gmmv(b1,x,p))
G<-jacobian(gmmm,b1,data=x)
vb1<-solve(t(G)%*%W%*%G)
b1
sqrt(diag(vb1))

rm(list = ls())
