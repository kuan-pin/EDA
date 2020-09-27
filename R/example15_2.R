# Example 15.2 ARMA Analysis of U. S. Inflation

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usinf.txt",header=T,nrows=136)

Y<-log(ts(data[,2],start=c(1950,1),frequency=4))
M<-log(ts(data[,3],start=c(1950,1),frequency=4))
P<-log(ts(data[,4],start=c(1950,1),frequency=4))

lagn<-function(x,l) {c(rep(NA,l),x[1:(length(x)-l)])}

dy<-100*(Y-lagn(Y,1))
dm<-100*(M-lagn(M,1))
dp<-100*(P-lagn(P,1))
dmy<-dm-dy

# time series analysis: identification
acf(dp,lag.max=12,type="correlation",na.action = na.pass,xlab="Lag (Qtr)") 
acf(dp,lag.max=12,type="partial",na.action = na.pass,xlab="Lag (Qtr)") 

# time series analysis: estimation
arma1<-arima(dp,order=c(1,0,3),xreg=dmy,method=c("ML"))
# check for serial correlation
Box.test(arma1$resid)
acf(arma1$resid,na.action=na.pass)

arma2<-arima(dp,order=c(0,0,3),xreg=cbind(dmy,lagn(dp,1)),method=c("ML"))
# check for serial correlation
Box.test(arma2$resid)
acf(arma2$resid,na.action=na.pass)

rm(list=ls())
