# Example 15.4: GARCH(1,1) Model of DM/BP Exchange Rate
# Bollerslerv and Ghysels [1996], JBES, 307-327.
#
# daily exchange rate data from 1/3/1984 to 12/31/1991 (obs.=1974)
dmbp<-read.table("http://web.pdx.edu/~crkl/ceR/data/dmbp.txt",nrows=1974)
xrate<-as.ts(dmbp[,1])
friday<-as.ts(dmbp[,2])

lm.2<-lm(xrate~friday)
summary(lm.2)

lm.1<-lm(xrate~1)
summary(lm.1)
e<-lm.1$residuals

for(j in 1:5) print(Box.test(e,lag=j,type="Box-Pierce"))
for(j in 1:5) print(Box.test(e,lag=j,type="Ljung-Box"))
acf(e,type="correlation")
acf(e,type="partial")

archlmtest <- function (x, lags, demean = FALSE) 
{ 
  x <- as.vector(x) 
  if(demean) x <- scale(x, center = TRUE, scale = FALSE) 
  lags <- lags + 1 
  mat <- embed(x^2, lags) 
  arch.lm <- summary(lm(mat[, 1] ~ mat[, -1])) 
  STATISTIC <- arch.lm$r.squared * length(resid(arch.lm)) 
  names(STATISTIC) <- "Chi-squared" 
  PARAMETER <- lags - 1 
  names(PARAMETER) <- "df" 
  PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER) 
  METHOD <- "ARCH LM-test" 
  result <- list(statistic = STATISTIC, parameter = PARAMETER, 
                 p.value = PVAL, method = METHOD, data.name = 
                   deparse(substitute(x))) 
  class(result) <- "htest" 
  return(result) 
} 
archlmtest(e,24)[1:3]

e2<-e^2
for(j in 1:5) print(Box.test(e2,lag=j,type="Box-Pierce"))
for(j in 1:5) print(Box.test(e2,lag=j,type="Ljung-Box"))
acf(e2,type="correlation")
acf(e2,type="partial")
shapiro.test(e2)

hist(e2,freq=F)
lines(density(e2))
qqnorm(e2)
qqline(e2)

library(rugarch)
mean1<-list(armaOrder=c(0,0),include.mean=TRUE)
var1<-list(model="sGARCH",garchOrder=c(1,1))
spec1<-ugarchspec(variance.model=var1,mean.model=mean1)
garch1<-ugarchfit(spec1,xrate)
garch1
plot(garch1)

rm(list=ls())
