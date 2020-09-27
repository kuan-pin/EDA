# Example 15.1 ARMA Analysis of Bond Yields

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/bonds.txt",header=T,nrows=60)
Y<-ts(data[,2],start=c(1990,1),frequency=4)

# time series analysis: identification
acf(Y,type="correlation",xlab="Lag (Year)") # acf(Y) => MA(2)
acf(Y,type="partial",xlab="Lag (Year)")     # pacf(Y) => AR(1)

# time series analysis: estimation
arma1<-arima(Y,order=c(2,0,0),method =c("ML"))  # AR(2)
arma1
# check for serial correlation
Box.test(arma1$resid)
acf(arma1$resid)

# alternative model
arma2<-arima(Y,order=c(1,0,1),method =c("ML"))  # ARMA(1,1)
arma2
# check for serial correlation
Box.test(arma2$resid)
acf(arma2$resid)

rm(list=ls())
