# Example 11.2 Lagged Dependent Variable Model
# Instrumental Variable Estimation
# Using package AER (inc. car, sandwich)

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",header=T,nrows=66)

y<-ts(data$Y,start=1929,frequency=1)
c<-ts(data$C,start=1929,frequency=1)
# define lag operator L on one variable
L<-function(x,l) {c(rep(NA,l),x[1:(length(x)-l)])}

library(AER)  # call ivreg
dlag.iv<-ivreg(c~y+L(c,1)|y+L(y,1)+L(y,2))
summary(dlag.iv)

# difficult to combine AR(1) and IV estimation
rm(list=ls())
