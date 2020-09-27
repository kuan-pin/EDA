# Example 6.2
# Two-Variable Scalar-Valued Function

setwd("C:/Course17/ceR/R")

# 2-variable scalar-valued function f(x):
f<-function(x) {
  x1<-x[1]
  x2<-x[2]
  return((x1^2+x2-11)^2+(x1+x2^2-7)^2)
}

# 1st analytic derivative of f(x)
f1<-function(x) {
  x1<-x[1]
  x2<-x[2]
  fx1<-4*x1*(x1^2+x2-11)+2*(x1+x2^2-7);
  fx2<-2*(x1^2+x2-11)+4*x2*(x1+x2^2-7);
  return(c(fx1,fx2))
}

# 2nd analytic derivative of f
f2<-function(x) {
  x1<-x[1]
  x2<-x[2]
  fx11=12*x1^2+4*x2-42;
  fx22=4*x1+12*x2^2-26;
  fx12=4*(x1+x2);
  return(matrix(c(fx11,fx12,fx12,fx22),2,2))
}

# function evaluation at point v
v<-c(1,1)
f(v)    
f1(v)
f2(v)

# Numeric Derivatives
library(numDeriv)          # need to install package numDeriv
grad(f,v)                  # 1st numeric derivative of f(x)
hessian(f,v)               # 2nd numeric derivative of f(x)

# find 4 minima of f(x,y) at: f1=0 0, f2= positive definite
# (3,2), (3.5844,-1.8481), (-3.7793,-3.2832), (-2.8051,3.1313)
# there is 1 maximum: (-0.27084,-0.92304) with f=181.62
# saddle points: (0.08668,2.88430), (3.38520,0.07358), (-3.07300,-0.08135)

# Using graph to find the minima
# persp function for perspective surface plot
# contour function for contour plot
x1<-seq(-5,5,length.out=81)
x2<-seq(-5,5,length.out=81)
xv<-as.matrix(expand.grid(x1,x2))
fv<-function(x) {
  x1<-x[,1]
  x2<-x[,2]
  return((x1^2+x2-11)^2+(x1+x2^2-7)^2)
}
yv<-fv(expand.grid(x1,x2))
y<-matrix(yv,81,81)   # surface matrix
persp(x1,x2,y)        # surface plot
persp(x1,x2,y,theta=45,phi=30,ticktype="detailed")
contour(x1,x2,y)      # contour plot
contour(x1,x2,y,nlevels=100)

# use nlm function to find 4 minma (see example2_1.R)
min1<-nlm(f,c(1,1))
min2<-nlm(f,c(2,-3))
min3<-nlm(f,c(-2,3))
min4<-nlm(f,c(-2,-3))
points(x=min1$estimate[1],y=min1$estimate[2],col="red",pch=20)
points(x=min2$estimate[1],y=min2$estimate[2],col="red",pch=20)
points(x=min3$estimate[1],y=min3$estimate[2],col="red",pch=20)
points(x=min4$estimate[1],y=min4$estimate[2],col="red",pch=20)

# use optim function to find local maximum
max1<-optim(c(-0.25,-1),f,method="BFGS",control=list(fnscale=-1))
max1$par
max1$value
points(x=max1$par[1],y=max1$par[2],col="blue",pch=20)

rm(list=ls())
