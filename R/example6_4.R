# Example 6.4
# Mixture of Two Normal Distributions

setwd("C:/Course17/ceR/R")
# Estimating a mixture of probability functions
# for a given set of data (sample) following a probability distribution
# find the parameters to maximize the probabity (likelihood)
#
yed<- read.table("http://web.pdx.edu/~crkl/ceR/data/yed20.txt",header=T,nrows=20)
# file name may be case sensitive!
# yed<- read.table("C:/course17/ec575/data/yed20.txt",header=T,nrows=20)
summary(yed)
y=yed$Income/10  # scale the data may help, y ~ prob. dist.

# mixture of two normal distributions
# b[1]=mu1, b[2]=sigma1, b[3]=mu2, b[4]=sigma2
llf1<-function(b) log(b[5]*dnorm(y,b[1],b[2])+(1-b[5])*dnorm(y,b[3],b[4]))
llf<-function(b) sum(llf1(b))

library(maxLik)  # need to install maxLik package
M1<-maxLik(llf,start=c(3,3,2,2,0.5))
summary(M1)

M1<-maxLik(llf1,start=c(3,3,2,2,0.5))
summary(M1)

# optim(c(3,2,2,1,0.5),llf,method="BFGS",hessian=T,control=list(fnscale=-1))
rm(list=ls())
