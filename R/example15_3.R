# Example 15.3 ARCH Analysis of U.S. Inflation

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usinf.txt",header=T,nrows=136)

y<-log(data$Y)
m<-log(data$M1)
p<-log(data$P)
dp<-100*diff(p,1)
dm<-100*diff(m,1)
dy<-100*diff(y,1)

ar3<-arima(dp,order=c(3,0,0))
ar3
e<-ar3$residuals
Box.test(e)
acf(e)

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

# time series analysis: diagnostic checking for volatility
e2<-e^2
acf(e2,type="correlation")
acf(e2,type="partial")
# check for normality of e^2
shapiro.test(e2)
qqnorm(e2)
qqline(e2)

library(rugarch)
var1<-list(model="sGARCH",garchOrder=c(1,1))
mean1<-list(armaOrder=c(3,0),include.mean=TRUE)
spec1<-ugarchspec(variance.model=var1,mean.model=mean1)
garch1<-ugarchfit(spec1,dp)
garch1

rm(list=ls())
