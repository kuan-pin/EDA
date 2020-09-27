# Example 4.1
# Seasonal Dummy Variables

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/almon.txt",header=T,nrows=60)

cexp <- data$CEXP
capp <- data$CAPP
# seasoanl factor variable
qt <- as.factor(data$YEARQT %% 10)

results<-lm(cexp~ capp+qt)
summary(results)

results1<-lm(cexp~ capp+qt-1)
summary(results1)

rm(list=ls())
