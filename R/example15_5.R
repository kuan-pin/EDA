# Example 15.5: GARCH(1,1) Model of DM/BP Exchange Rate
# Bollerslerv and Ghysels [1996], JBES, 307-327.
# Model Variations: IGARCH(1,1), GARCH-M(1,1), EGARCH(1,1)
# Considering Non-Gaussian Distribution: GED
#
# daily exchange rate data from 1/3/1984 to 12/31/1991 (obs.=1974)
dmbp<-read.table("http://web.pdx.edu/~crkl/ceR/data/dmbp.txt",nrows=1974)
xrate<-as.ts(dmbp[,1])
friday<-as.ts(dmbp[,2])

library(rugarch)

mean1<-list(armaOrder=c(0,0),include.mean=TRUE)
mean2<-list(armaOrder=c(0,0),include.mean=TRUE,archm=TRUE)
var1<-list(model="sGARCH",garchOrder=c(1,1))
var2<-list(model="iGARCH",garchOrder=c(1,1))
var3<-list(model="gjrGARCH",garchOrder=c(1,1))
var4<-list(model="eGARCH",garchOrder=c(1,1))
# including x-regressors in the mean or in the variance
mean1x<-list(armaOrder=c(0,0),include.mean=TRUE,external.regressors=as.matrix(friday))
var1x<-list(model="sGARCH",garchOrder=c(1,1),external.regressors=as.matrix(friday))

# GARCH(1,1)
spec11<-ugarchspec(variance.model=var1,mean.model=mean1)
garch11<-ugarchfit(spec11,xrate)
garch11
plot(garch11)

# GARCH-M(1,1)
spec12<-ugarchspec(variance.model=var1,mean.model=mean2)
garch12<-ugarchfit(spec12,xrate)
garch12

# IGARCH(1,1)
spec21<-ugarchspec(variance.model=var2,mean.model=mean1)
garch21<-ugarchfit(spec21,xrate)
garch21
plot(garch21)

# GJR-GARCH(1,1)
spec31<-ugarchspec(variance.model=var3,mean.model=mean1)
garch31<-ugarchfit(spec31,xrate)
garch31

# EGARCH(1,1)
spec41<-ugarchspec(variance.model=var4,mean.model=mean1)
garch41<-ugarchfit(spec41,xrate)
garch41

# try non-normal distributions: std, ged,...
spec11a<-ugarchspec(variance.model=var1,mean.model=mean1,distribution.model="ged")
garch11a<-ugarchfit(spec11a,xrate)  
garch11a
plot(garch11a)

spec41a<-ugarchspec(variance.model=var4,mean.model=mean1,distribution.model="ged")
garch41a<-ugarchfit(spec41a,xrate)
garch41a

# GARCHX(1,1)
spec11x<-ugarchspec(variance.model=var1x,mean.model=mean1x)
garch11x<-ugarchfit(spec11x,xrate)
garch11x
plot(garch11x)

rm(list=ls())
