# Example 8.3 Tobit Analysis of Extramarital Affairs
# Poisson Regression Model
# Analysis of Extramarital Affairs (Fair, 1978)

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/fair.txt",header=T,nrows=601)

y<-data$Y
z<-cbind(data$Z2,data$Z3,data$Z5,data$Z7,data$Z8)

# Tobit Model
# install.packages("AER")
library("AER")
model0 <- tobit(y~z)
summary(model0)

#Poisson Model
model1<-glm(y~z,poisson)  # default link=log
summary(model1)

# Alternatively using glm.nb()
library(MASS)  # need to install package MASS
model2<-glm.nb(y~z,link=log)  # default link=log
summary(model2)

rm(list=ls())

