# Example 17.2: Ex-Ante Forecasts
# Using package forecast

setwd("C:/Course17/ceR/R")
# read data from 1959.1 to 2003.4 (181 obs.)
z<-read.table("http://web.pdx.edu/~crkl/ceR/data/gdp96.txt",header=T)
gdp<-ts(z$GDP,start=c(1959,1),frequency=4)
pgdp<-ts(z$PGDP2000,start=c(1959,1),frequency=4)
# constant scenario (0% AGR)
leading<-ts(c(z$LEADING96,115.0,115.0,115.0,115.0,115.0),start=c(1959,1),frequency=4)
# pessimistic scenario (-2% AGR)
# leading<-ts(c(z$LEADING96,115.0,114.4,113.8,113.2,112.6),start=c(1959,1),frequency=4)
# constant scenario (0% AGR)
# leading<-ts(c(z$LEADING96,115.0,115.0,115.0,115.0,115.0),start=c(1959,1),frequency=4)
# optimistic scenario (+2% AGR)
# leading<-ts(c(z$LEADING96,115.0,115.6,116.2,116.8,117.4),start=c(1959,1),frequency=4)
rgdp<-100*gdp/pgdp
growth<-100*(rgdp-lag(rgdp,-4))/lag(rgdp,-4)
xvar<-ts.intersect(lag(leading,-1),lag(leading,-5))  

growth.train<-window(growth,start=c(1961,1),end=c(2001,4))
xvar.train<-window(xvar,start=c(1961,1),end=c(2001,4))
growth.test<-window(growth,start=c(2002,1))
xvar.test<-window(xvar,start=c(2002,1))

arma1<-arima(growth.train,order=c(1,0,4),xreg=xvar.train)
arma1
for1<-predict(arma1,n.ahead=14,newxreg=xvar.test)
for1
plot(for1$pred)
lines(growth.test,lty=2)
lines(for1$pred+for1$se,col="red")
lines(for1$pred-for1$se,col="red")

library(forecast)
arma2<-Arima(growth.train,order=c(1,0,4),xreg=xvar.train)
arma2
for2<-forecast(arma2,h=14,xreg=xvar.test)
for2
accuracy(for2,growth.test)
plot(for2)
plot(for2,include=10)
lines(growth.test)

