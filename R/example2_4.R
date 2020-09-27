# Example 2.4
# Data Analysis
setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

y<-as.matrix(data[,7])
#x<-as.matrix(cbind(rep(1,nrow(data)),data[,1:6]))
x<-as.matrix(cbind(1,data[,1:6]))
summary(x)
xx<-t(x)%*%x
xy<-t(x)%*%y
invxx<-solve(xx)
b<-invxx%*%xy
b

cm<-eigen(xx)$val
cn<-sqrt(cm[1]/cm[7])
cond<-kappa(x,exact = TRUE)
cat(cn,cond)  # c(cn,cond)

rm(list=ls())

