# Example 11.3 Polynomial Distributed Lag Model
# Almon Lag (Lags=7, Order=4)
# Almon Lag (Lags=7, Order=4 End=2)+Seasonality: difficult, not estimated, but tested

almon<-read.table("http://web.pdx.edu/~crkl/ceR/data/almon.txt",header=T,nrows=60)
cexp<-ts(almon$CEXP,start=c(1953,1),end=c(1961,4),frequency=4)
capp<-ts(almon$CAPP,start=c(1953,1),end=c(1961,4),frequency=4)  
d<-as.factor(cycle(cexp))  # seasonality

pdl0<-lm(cexp~capp+d-1)
summary(pdl0)

# define lag operator L on one variable
L<-function(x,l) {c(rep(NA,l),x[1:(length(x)-l)])}

H<-function(q,p) {
  Hmat<-matrix(nrow=q,ncol=p)
  for(i in 1:q) {
    for(j in 1:p) {Hmat[i,j]<-i^j}
  }
  Hmat<-rbind(rep(0,p),Hmat)
  Hmat<-cbind(rep(1,q+1),Hmat)
  Hmat
}

X<-cbind(capp,L(capp,1),L(capp,2),L(capp,3),L(capp,4),L(capp,5),L(capp,6),L(capp,7))
H<-H(7,4)
Z<-X%*%H

# unrestricted 7-lags model
pdl1<-lm(cexp~X+d-1)
summary(pdl1)

# polynomial lags: restricted, lags=7 order=4
pdl2<-lm(cexp~Z+d-1)
summary(pdl2)

H%*%pdl2$coefficients[1:5]
v<-vcov(pdl2)
diag(H%*%v[1:5,1:5]%*%t(H))

# check for end-point restrictions
library(car)
lht(pdl2,"Z1-Z2+Z3-Z4+Z5=0")               # left-end restriction
lht(pdl2,"Z1+8*Z2+64*Z3+512*Z4+4096*Z5=0") # right-end restriction

# check for autocorrelation
dwt(pdl2)

rm(list = ls())
