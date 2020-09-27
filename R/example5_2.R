# Example 5.2
# Theil's Measure of Multicollinearity

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/longley.txt",header=T,nrows=16)

year<-data$YEAR
pgnp<-data$PGNP
gnp<-data$GNP
af<-data$AF
em<-data$EM

results<-lm(em~year+pgnp+gnp+af)
results_summary<-summary(results)
results_summary
R2<-results_summary[8]

results<-lm(em~pgnp+gnp+af)
results_summary<-summary(results)
results_summary
R21<-results_summary[8]

results<-lm(em~year+gnp+af)
results_summary<-summary(results)
results_summary
R22<-results_summary[8]

results<-lm(em~year+pgnp+af)
results_summary<-summary(results)
results_summary
R23<-results_summary[8]

results<-lm(em~year+pgnp+gnp)
results_summary<-summary(results)
results_summary
R24<-results_summary[8]

print ("Theil's Measure of Multicollinearity =")

R2<-as(R2,"numeric")
R21<-as(R21,"numeric")
R22<-as(R22,"numeric")
R23<-as(R23,"numeric")
R24<-as(R24,"numeric")

R<-R2-((R2-R21)+(R2-R22)+(R2-R23)+(R2-R24))
R

rm(list=ls())
