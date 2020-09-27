# Example 17.1: Ex-Post Forecasts and Forecast Error Statistics

setwd("C:/Course17/ceR/R")
# read data from 1959.1 to 2003.4 (181 obs.)
z<-read.table("http://web.pdx.edu/~crkl/ceR/data/gdp96.txt",header=T)
gdp<-ts(z$GDP,start=c(1959,1),frequency=4)
pgdp<-ts(z$PGDP2000,start=c(1959,1),frequency=4)
leading<-ts(z$LEADING96,start=c(1959,1),frequency=4)

rgdp<-100*gdp/pgdp
growth<-100*(rgdp-lag(rgdp,-4))/lag(rgdp,-4)
xvar<-ts.intersect(lag(leading,-1),lag(leading,-5))  

growth.train<-window(growth,start=c(1961,1),end=c(2001,4))
xvar.train<-window(xvar,start=c(1961,1),end=c(2001,4))
growth.test<-window(growth,start=c(2002,1))
xvar.test<-window(xvar,start=c(2002,1))

arma1<-arima(growth.train,order=c(1,0,4),xreg=xvar.train)
arma1
for1<-predict(arma1,n.ahead=9,newxreg=xvar.test)
for1
plot(for1$pred)
lines(for1$pred+for1$se,col="red")
lines(for1$pred-for1$se,col="red")

# write a function to compute error statistics
predict.error<-function(x,p) {
  data<-ts.intersect(x,p)
  x<-data[,"x"]
  p<-data[,"p"]
  e<-x-p                           # prediction error
  mx<-mean(x)
  mp<-mean(p)
  sx<-sqrt(mean((x-mx)^2))
  sp<-sqrt(mean((p-mp)^2))
  r<-cor(x,p)
  MSE<-mean(e^2)                   # mean squared error
  # results list
  results=list()
  results$r2<-r^2
  results$ME<-mean(e)                      # mean error
  results$MAE<-mean(abs(e))                # mean absolute error
  results$MAPE<-100*mean(abs(e/x))         # mean absolute error
  results$RMSE<-sqrt(MSE)                  # root mean squared error
  results$RMSPE<-100*sqrt(mean((e/x)^2))   # root mean squared error
  # MSE Decomposition
  results$MSE<-MSE
  results$Um<-((mx-mp)^2)/MSE
  results$Us<-((sx-sp)^2)/MSE
  results$Uc<-(2*(1-r)*sp*sx)/MSE
  results$Ur<-((sp-r*sx)^2)/MSE
  results$Ud<-((1-r^2)*sx^2)/MSE
  return(results)
}
predict.error(growth.test,for1$pred)

