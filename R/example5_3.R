# Example 5.3
# Lesson 5.3: Variance Inflation Factors (VIF)

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

year<-data$YEAR
pgnp<-data$PGNP
gnp<-data$GNP
af<-data$AF
em<-data$EM

results<-lm(em~year+pgnp+gnp+af)
summary(results)

library(car)
vif(results)

sqrt(vif(results))>2    # Whether it has Multicollinearity

rm(list=ls())
