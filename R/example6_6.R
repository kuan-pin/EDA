# Example 6.6 Maximizing Log-Likelihood Function
# Estimating a CES Production Function

setwd("C:/Course17/ceR/R")
# Estimating a CES Production Function
# Judge, et. al. [1988], Chapter 12
judge<- read.table("http://web.pdx.edu/~crkl/ceR/data/judge.txt")

summary(judge)

# objective function to be maximized
llf1<-function(b) {
  l<-judge[,1]
  k<-judge[,2]
  q<-judge[,3]
  e<-log(q)-b[1]-b[4]*log(b[2]*l^b[3]+(1-b[2])*k^b[3])
  ll<-dnorm(e,mean=0,sd=sqrt(mean(e^2)),log=T)
  ll<-ll+log(1/q)  # add log(jacobian)
  return(ll)
}
llf<-function(b) sum(llf1(b))

# Maximum Likelihood
library(maxLik)   # need to install maxLik package
M3<-maxLik(llf1,start=c(1,0.5,-1,-1))
summary(M3)
M3$maximum
M3$estimate
M3$gradient
M3$hessian
# compute var-cov matrix for se(b)        
G<-t(M3$gradientObs)%*%M3$gradientObs
H<-M3$hessian
se1<-sqrt(diag(solve(G)))
se2<-sqrt(diag(solve(-H)))
se3<-sqrt(diag(solve(H)%*%G%*%solve(H)))
M3$estimate
se1 # se1=se2 or -H=G if Information Matrix Equality holds
se2 # standard errors
se3 # robust standard errors

rm(list=ls())


