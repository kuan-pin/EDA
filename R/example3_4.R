# Example 3.4
# Cobb-Douglas Production Function

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year<-data$YEAR
X<-log(data$X)
L<-log(data$L1) 
K<-log(data$K1)

results<-lm(X ~ L+K)
summary(results)
anova(results)

# constant returns to scale: restricted least squares
# linear restrictions via parameter subsititution
results1<-lm(X ~ I(L-K),offset=K)
summary(results1)
anova(results1)

# F test for linear restrictions
anova(results1,results)

# using package car
library(car)
linearHypothesis(results,"L+K=1")
# lht(results,"L+K=1")

rm(list=ls())
