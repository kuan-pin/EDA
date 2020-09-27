# Example 7.1 CES Production Function Revisited
# Estimating CES Production Function
# Judge, et. al. [1988], Chapter 12

setwd("C:/Course17/ceR/R")
judge<- read.table("http://web.pdx.edu/~crkl/ceR/data/judge.txt")
summary(judge)

# Nonlinear least squares
ces<-log(V3)~(b1+b4*log(b2*V1^b3+(1-b2)*V2^b3))
M0<-nls(ces,data=judge,start=list(b1=1,b2=0.5,b3=-1,b4=-1))
summary(M0)

rm(list=ls())
