# Example 14.2 Cointegration Test
# Johansen Approach
# Using package urca

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",header=T,nrows=66)
Y<-ts(data[,2],start=1929)
C<-ts(data[,3],start=1929)
X<-cbind(C,Y)

library(urca)
# Johansen Cointegration test: eigenvalue statistic
test.jo3<-ca.jo(X,type="eigen",ecdet="trend",K=3,spec="transitory")
summary(test.jo3)

test.jo2<-ca.jo(X,type="eigen",ecdet="const",K=3,spec="transitory")
summary(test.jo2)

test.jo1<-ca.jo(X,type="eigen",K=3,spec="transitory")
summary(test.jo1)

# Johansen Cointegration test: trace statistic
test.jo3<-ca.jo(X,"trace","trend",K=3,spec="transitory")
summary(test.jo3)

test.jo2<-ca.jo(X,"trace","const",K=3,spec="transitory")
summary(test.jo2)

test.jo1<-ca.jo(X,"trace",K=3,spec="transitory")
summary(test.jo1)

rm(list=ls())
