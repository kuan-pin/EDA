# Example 9.2 Goldfeld-Quandt Test and Correction for Heteroscedasticity

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/greene.txt",header=T,nrows=51)
data<-na.omit(data)  # take care of missing obs
data1<-data[order(data[,3]),]

spending<-data1$SPENDING
income<-data1$INCOME/10000

results<-lm(spending~income+I(income^2))
summary(results)

# Goldfeld-Quandt test
results1<-lm(spending~income+I(income^2),subset=1:17 )
summary(results1)
rss1<-sum(results1$resid^2)
df1<-results1$df.residual

results2<-lm(spending~income+I(income^2),subset=35:51 )
summary(results2)
rss2<-sum(results2$resid^2)
df2<-results2$df.residual
GQstat<-(rss2/df2)/(rss1/df1)  # F-test
GQstat
1-pf(GQstat,df1,df2)

# weighted least squares
result.wls<-lm(spending~income+I(income^2),weights=1/income)
summary(result.wls)

rm(list=ls())
