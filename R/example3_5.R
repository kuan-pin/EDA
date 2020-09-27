# Example 3.5
# Testing for Structural Change

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year<-data$YEAR
X<-log(data$X)
L<-log(data$L1) 
K<-log(data$K1)

results<-lm(X ~ L+K)
summary(results)
results1<-lm(X~ L+K,subset=year<1949)
summary(results1)
results2<-lm(X~ L+K,subset=year>1948)
summary(results2)

## Calculate RSS and DF for restricted and unrestricted models
RSSr<-sum(results$residuals^2)
DFr<-results$df.residual
RSSur<-sum(results1$residuals^2)+sum(results2$residuals^2)
DFur<-results1$df.residual+results2$df.residual

## Computing the Chow test statistic (F-test)
Ftest<-((RSSr-RSSur)/(DFr-DFur))/(RSSur/DFur)
Ftest

## Calculate P-value
1-pf(Ftest, DFr-DFur, DFur)

rm(list=ls())
