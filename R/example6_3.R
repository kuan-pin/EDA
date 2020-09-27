# Example 6.3
# Estimating Probability Distributions
# for a given set of data (sample) following a probability distribution
# find the parameters to maximize the probabity (likelihood)

setwd("C:/Course17/ceR/R")
yed<- read.table("http://web.pdx.edu/~crkl/ceR/data/yed20.txt",header=T,nrows=20)
# file name may be case sensitive!
# yed<- read.table("C:/course17/ec575/data/yed20.txt",header=T,nrows=20)
summary(yed)
y=yed$Income/10  # scale the data may help, y ~ prob. dist.

# normal likelihood: b[1]=mu, b[2]=sigma
nllf<-function(b) sum(dnorm(y,b[1],b[2],log=T))

# log-normal likelihood
lnllf<-function(b) sum(dlnorm(y,b[1],b[2],log=T))
# lnllf<-function(b) sum(dnorm(log(y),b[1],b[2],log=T)+log(1/y))

# gamma likelihood: b[1]=rho, b[2]=lambda (note: shape=rho, scale=1/lambda)
gllf<-function(b) sum(dgamma(y,shape=b[1],scale=1/b[2],log=T))
# gllf<-function(b) sum((b[1]*log(b[2])-lgamma(b[1]))-b[2]*y+(b[1]-1)*log(y))  

library(maxLik)  # need to install maxLik package
M1<-maxLik(nllf,start=c(3,2))
M2<-maxLik(lnllf,start=c(1,1))
M3<-maxLik(gllf,start=c(2,1))
summary(M1)
summary(M2)
summary(M3)

# optim(c(3,2),nllf,method="BFGS",hessian=T,control=list(fnscale=-1))
# optim(c(1,1),lnllf,method="BFGS",hessian=T,control=list(fnscale=-1))
# optim(c(2,1),gllf,method="BFGS",hessian=T,control=list(fnscale=-1))

rm(list=ls())
