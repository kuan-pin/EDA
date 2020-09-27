# Example 6.1
# One-Variable Scalar-Valued Function

setwd("C:/Course17/ceR/R")

f<-function(x) log(x)-x^2  # 1-variable scalar-valued function
v<-0.5          			     # function evaluation at point v=0.5
f(v)                       # f(0.5)

# Numeric Derivatives
library(numDeriv)          # need to install package numDeriv
grad(f,v)                  # 1st numeric derivative of f(x)
hessian(f,v)               # 2nd numeric derivative of f(x)

# Using graph to find the maximum of f
x<-seq(0.0001,4,length.out=40)
y<-f(x)
plot(x,y)
plot(x,y,type="l",ylim=c(-15,0),xlim=c(0,4))

# find maximum of f(x) at x=sqrt(0.5)
max1<-optim(v,f,method="BFGS",control=list(fnscale=-1))
max1$par
max1$value
points(x=max1$par,y=max1$value,col="red",pch=20)

# or, using nlm
fn<-function(x) -f(x)      # negative f, to find minimum of fn
nlm(fn,v)

rm(list=ls())

