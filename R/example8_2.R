# Example 8.2 Logit Model of Economic Education
# Binomial Regression Model: Probit and Logit

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/grade.txt",header=T,nrows=32)

gpa<-data$GPA
tuce<-data$TUCE
psi<-data$PSI
grade<-data$GRADE

model1<-glm(grade~gpa+tuce+psi,binomial)  # default link=logit
summary(model1)

# logit model interpretation
b1<-model1$coefficients
# estimated probability
probility<-predict(model1,type="response")       

# marginal effects of an explanatory variable
gpa_slopes<-probility*(1-probility)*b1["gpa"]
tuce_slopes<-probility*(1-probility)*b1["tuce"]
psi_slopes<-probility*(1-probility)*b1["psi"]

cbind(grade,probility,gpa_slopes,tuce_slopes,psi_slopes)

rm(list=ls())
