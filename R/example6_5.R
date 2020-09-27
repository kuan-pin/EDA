# Example 6.5 Minimizing Sum-of-Squares Function
# Estimating a CES Production Function

setwd("C:/Course17/ceR/R")
# Estimating a CES Production Function
# Judge, et. al. [1988], Chapter 12
# 
judge<- read.table("http://web.pdx.edu/~crkl/ceR/data/judge.txt")

summary(judge)

# objective function to be minimized
sse<-function(b) {
  l<-judge[,1]
  k<-judge[,2]
  q<-judge[,3]
  e<-log(q)-b[1]-b[4]*log(b[2]*l^b[3]+(1-b[2])*k^b[3])
  return(sum(e^2))
}

M1<-optim(c(1,0.5,-1,-1),sse,method="BFGS",hessian=T)
v<-(M1$value/nrow(judge))
var<-diag(v*solve(0.5*M1$hessian))
M1$hessian
M1$par
sqrt(var)
M1$value

rm(list=ls())

