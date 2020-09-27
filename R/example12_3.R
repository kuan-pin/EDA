# Example 12.3 GMM Estimation of U.S. Consumption Function
# Using package AER, gmm

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",header=T,nrows=66)
y<-ts(data$Y,start=1929,frequency=1)
c<-ts(data$C,start=1929,frequency=1)

# define lag operator L on one variable
L<-function(x,l) {c(rep(NA,l),x[1:(length(x)-l)])}

results.ols<-lm(c~L(c,1))
summary(results.ols)

library(AER)
# update/reinstall the following packages: rio,curl,data.table,magritter,hms,
# pkgconfig,crayon,stringi,zip,readxl,cellranger,abind,lmtest,sandwich,Formula
results.iv<-ivreg(c~L(c,1)|L(y,1))
summary(results.iv)

library(gmm)
# can npot run the following?
model1<-gmm(c~L(c,1),~L(y,1),results.ols$coefficient)  # default vcov="HAC"
summary(model1)
y1<-L(y,1)
c1<-L(c,1)
cc<-c[2:66]
c1<-c1[2:66]
y1<-y1[2:66]
yy<-y[2:66]
model1<-gmm(cc~c1,~y1,results.ols$coefficients)  # default vcov="HAC"
summary(model1)
model2<-gmm(cc~c1,~y1,results.ols$coefficients,type="iterative")
summary(model2)
model3<-gmm(cc~c1,~y1,results.ols$coefficients,type="cue")
summary(model3)

rm(list=ls())
