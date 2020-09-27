# Example 5.4
# Ridge Regression and Principal Components

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

year<-data$YEAR
pgnp<-data$PGNP
gnp<-data$GNP
af<-data$AF
em<-data$EM

results<-lm(em~year+pgnp+gnp+af)
summary(results)
b<-results$coefficients

library(car)
library(MASS)

#Ridge Regression Model
ridgelm<-lm.ridge(em~year+pgnp+gnp+af)
ridgelm$coef

#install.packages("ridge")
#library(ridge)
#mod <- linearRidge(em~year+pgnp+gnp+af)
#summary(mod)

# plot
plot(lm.ridge(em~year+pgnp+gnp+af,lambda = seq(0,0.1,0.001)))

#Principal Components Model
x<-cbind(1,year,pgnp,gnp,af)
xx<-t(x)%*%x
M<-eigen(xx)
V<-M$vectors[1:5,5]
b<-as.matrix(c(b))
b1<-V%*%t(V)%*%b
b1

rm(list=ls())
