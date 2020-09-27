# Example 2.2
# File I/O

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

PGNP <- data[,2]
GNP <- data[,3]/1000   
POPU <- data[,6]/1000
EM <- data[,7]/1000

x=cbind(PGNP,GNP,POPU,EM)
x

rm(list=ls())
