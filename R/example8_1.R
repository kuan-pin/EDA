# Example 8.1 Probit Model of Economic Education
# Binomial Regression Model: Probit and Logit

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/grade.txt",header=T,nrows=32)

gpa<-data$GPA
tuce<-data$TUCE
psi<-data$PSI
grade<-data$GRADE

model2<-glm(grade~gpa+tuce+psi,binomial(link=probit))  # link=probit
summary(model2)

# probit model interpretation
b2<-model2$coefficients
probility<-predict(model2,type="response")       # estimated probability

# marginal effects of an explanatory variable
x<-cbind(1,gpa,tuce,psi)
gpa_slopes<-dnorm(x%*%b2,mean=0,sd=1)*b2["gpa"]
tuce_slopes<-dnorm(x%*%b2,mean=0,sd=1)*b2["tuce"]
psi_slopes<-dnorm(x%*%b2,mean=0,sd=1)*b2["psi"]

cbind(grade,probility,gpa_slopes,tuce_slopes,psi_slopes)

rm(list=ls())
