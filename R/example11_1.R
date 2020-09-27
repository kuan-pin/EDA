# Example 11.1 Lagged Dependent Variable Model
# Using package lmtest, orcutt

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",header=T,nrows=66)
y<-ts(data$Y,start=1929,frequency=1)
c<-ts(data$C,start=1929,frequency=1)

# define lag operator L on one variable
L<-function(x,l) {c(rep(NA,l),x[1:(length(x)-l)])}

c1<-L(c,1)
results<-lm(c~y+c1)
summary(results)
VB<-vcov(results)
N<-length(results$resid)

library(lmtest)
# Durbin-Watson test
DW<-dwtest(results)
DW
# Durbin-H test
DH<-(1-DW$statistic/2)*sqrt(N/(1-N*VB["c1","c1"]))
DH; 1-pnorm(DH)

library(orcutt)
results.co <- cochrane.orcutt(results)
results.co
for(i in 1:4) print(bgtest(results.co[[1]],i))

rm(list=ls())
