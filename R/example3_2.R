# Example 3.2
# Residual Analysis

data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

PGNP<-data$PGNP
GNP<-data$GNP/1000
EM<-data$EM/1000
RGNP<-100*GNP/PGNP

results<-lm(EM~RGNP)
summary(results)
anova(results)

Predicted<-results$fitted.values
Residual<-results$residuals    

cbind(EM,Predicted,Residual)

old.par<-par(mfrow = c(2, 2), pty = "m") 
plot(results)

par(old.par)  # reset to old graph parameters
rm(list=ls())
