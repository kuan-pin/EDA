# Example 16.1 One-Way Panel Data Analysis, Dummy Variable
# Cost of Production for Airline Services I

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/airline.txt",header=T,nrows=90)

D <- as.factor(data$I)
cs<-log(data$C)
qs<-log(data$Q)
pfs<-log(data$PF)
lfs<-data$LF  # load factor, not logged

results<-lm(cs~qs+pfs+lfs+D)
summary(results)

results1<-lm(cs~qs+pfs+lfs)
summary(results1)
anova(results1,results)

rm(list=ls())
