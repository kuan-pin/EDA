# Example 5.1
# Condition Number and Correlation Matrix

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

year<-data$YEAR
pgnp<-data$PGNP
gnp<-data$GNP
af<-data$AF
em<-data$EM

results<-lm(em~year+pgnp+gnp+af)
summary(results)

#  Correlation Matrix
y<-cbind(year,pgnp,gnp,af,em)
y1<-cor(y)
y1

# Condition Number
x<-cbind(1,year,pgnp,gnp,af)
x1<-t(x)%*%x
cm<-eigen(x1)$val
cn<-sqrt(max(cm)/min(cm))
cn
kappa(x) # 2-norm condition number 

rm(list=ls())
