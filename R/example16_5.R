# Example 16.5 Panel Data Analysis for Investment Demand Function
# Seemingly Unrelated Regression Estimation
# Using package systemfit

# read 5 data files
data_1<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcgm.txt",header=TRUE,nrows=20)
data_2<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcch.txt",header=TRUE,nrows=20)
data_3<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcge.txt",header=TRUE,nrows=20)
data_4<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcwe.txt",header=TRUE,nrows=20)
data_5<- read.table("http://web.pdx.edu/~crkl/ceR/data/ifcus.txt",header=TRUE,nrows=20)

data1<- data_1
data2<- data_2[,-1]
data3<- data_3[,-1]
data4<- data_4[,-1]
data5<- data_5[,-1]

data<-cbind(data1,data2,data3,data4,data5)
names(data)<-c("YEAR","I_GM","F_GM","C_GM","I_CH",
               "F_CH","C_CH","I_GE","F_GE","C_GE",
               "I_WE","F_WE","C_WE","I_US","F_US","C_US")

formulas<-c(I_GM ~ F_GM + C_GM, I_CH ~ F_CH + C_CH, 
             I_GE ~ F_GE + C_GE, I_WE ~ F_WE+ C_WE, I_US ~ F_US + C_US)

library(systemfit)
# Perform a SUR Estimation
SUR<-systemfit(formulas,method = "SUR",data = data)
summary(SUR)

rm(list=ls())
