# Example 4.2
# Dummy Variable Trap

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/almon.txt",header=T,nrows=60)

cexp <- data$CEXP
capp <- data$CAPP
# seasoanl factor variable
qt <- as.factor(data$YEARQT %% 10)

results<-lm(cexp~ capp+qt)
summary(results)

rm(list=ls())
