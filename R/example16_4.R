# Example 16.4 Panel Data Analysis for Investment Demand
# Deviation Approach
# Using package plm

# read 5 data files
data_1<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcgm.txt",header=TRUE,nrows=20)
data_2<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcch.txt",header=TRUE,nrows=20)
data_3<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcge.txt",header=TRUE,nrows=20)
data_4<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcwe.txt",header=TRUE,nrows=20)
data_5<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcus.txt",header=TRUE,nrows=20)

data1<-cbind(1,data_1)
names(data1)<-c("ID","YEAR","I","F","C")
data2<-cbind(2,data_2)
names(data2)<-c("ID","YEAR","I","F","C")
data3<-cbind(3,data_3)
names(data3)<-c("ID","YEAR","I","F","C")
data4<-cbind(4,data_4)
names(data4)<-c("ID","YEAR","I","F","C")
data5<-cbind(5,data_5)
names(data5)<-c("ID","YEAR","I","F","C")

data<-rbind(data1,data2,data3,data4,data5)

library(plm)
# Set data as panel data
pdata <- pdata.frame(data, index=c("ID","YEAR"))

# Pooled OLS estimator
pooling <- plm(I ~ F+C, data=pdata, model= "pooling")
summary(pooling)

# Between estimator
between <- plm(I ~ F+C, data=pdata, model= "between")
summary(between)

# First differences estimator
firstdiff <- plm(I ~ F+C, data=pdata, model= "fd")
summary(firstdiff)

# Fixed effects or within estimator
fixed <- plm(I ~ F+C, data=pdata, model= "within")
summary(fixed)

# Random effects estimator
random <- plm(I ~ F+C, data=pdata, model= "random") # use random.method="swar"
summary(random)

# Pooled or fixed
pFtest(fixed,pooling)

# Pooled or random
plmtest(random,type="bp")

# Hausman test for fixed or random effects model
phtest(random,fixed)

rm(list=ls())

