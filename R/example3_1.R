# Example 3.1
# Simple Regression

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

PGNP<-data$PGNP
GNP<-data$GNP/1000
EM<-data$EM/1000
RGNP<-100*GNP/PGNP

results<-lm(EM~RGNP)
summary(results)
anova(results)

rm(list=ls())
