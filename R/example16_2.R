# Example 16.2 One-Way Panel Data Analysis, Deviation Approach
# Production of Airline Services: C = f(Q,PF,LF)
# Panel data: 6 airline companies, 15 years (1970-1984)
# Fixed effects and random effects models
# Using package plm, lmtest, car

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/airline.txt",header=T,nrows=90)
summary(data)

library(plm)
# Set data as panel data
pdata <- pdata.frame(data, index=c("I","T"))

# variable transfermation
Y<-log(pdata$C)
X<-cbind(log(pdata$Q),log(pdata$PF),pdata$LF)

# Descriptive statistics
summary(cbind(Y,X))

# F test for fixed effects versus OLS
pFtest(Y ~ X, data=pdata, effect="individual")
pFtest(Y ~ X, data=pdata, effect="time")
pFtest(Y ~ X, data=pdata, effect="twoways")

# LM test for random effects versus OLS
plmtest(Y ~ X, data=pdata, effect="individual", type="bp")
plmtest(Y ~ X, data=pdata, effect="time", type="bp")
plmtest(Y ~ X, data=pdata, effect="twoways", type="bp")

# Pooled OLS estimator
pooling <- plm(Y ~ X, data=pdata, model= "pooling")
summary(pooling)

# Between estimator
between <- plm(Y ~ X, data=pdata, model= "between")
summary(between)

# First differences estimator
firstdiff <- plm(Y ~ X, data=pdata, model= "fd")
summary(firstdiff)

# Fixed effects or within estimator
fixed <- plm(Y ~ X, data=pdata, model= "within")
summary(fixed)
# extract fixed effects
Y1<-Between(Y)
X1<-cbind(Between(log(pdata$Q)),Between(log(pdata$PF)),Between(pdata$LF))
Y1-X1%*%fixed$coefficients
summary(fixef(fixed))

# Random effects estimator
random <- plm(Y ~ X, data=pdata, model= "random") # use random.method="swar"
summary(random)
# extract random effects
ercomp(random)$theta*(Y1-cbind(1,X1)%*%random$coefficients)+random$coefficients[1]

# Hausman test for fixed versus random effects model
phtest(random,fixed)

library("lmtest")
coeftest(fixed,vcovHC)
coeftest(random,vcovHC)
#
# alternative hausman test based on re
# include group means in the random effects model
# test the significance of group mean coefficients
X1<-cbind(Between(log(pdata$Q)),Between(log(pdata$PF)),Between(pdata$LF))

# Random effects estimator
random1 <- plm(Y ~ X + X1, data=pdata, model= "random", random.method = "nerlove")
summary(random1)

library(car)
lht(random1,c("X11","X12","X13"))

rm(list=ls())
