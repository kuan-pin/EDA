# Example 16.3 Two-Way Panel Data Analysis, Dummy Variable
# Cost of Production for Airline Services II

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/airline.txt",header=T,nrows=90)

D <- as.factor(data$I)
T <- as.factor(data$T)

cs<-log(data$C)
qs<-log(data$Q)
pfs<-log(data$PF)
lfs<-data$LF

results<-lm(cs~qs+pfs+lfs+D+T)
summary(results)

results1<-lm(cs~qs+pfs+lfs+D)
summary(results1)
anova(results1,results)

results2<-lm(cs~qs+pfs+lfs+T)
summary(results2)
anova(results2,results)

results3<-lm(cs~qs+pfs+lfs)
summary(results3)

anova(results3,results2)
anova(results3,results1)
anova(results3,results)

rm(list=ls())
