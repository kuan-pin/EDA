# Example 2.3
# Data Transformation

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)


PGNP <- log(data[,2])
GNP <- log(data[,3]/1000)   
POPU <- log(data[,6]/1000)
EM <- log(data[,7]/1000)

x=cbind(PGNP,GNP,POPU,EM)
x

rm(list=ls())
