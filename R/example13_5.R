# Example 13.5 Klein's Model I Revisited
# Nonliear FIML Estimation, Goldfeld-Quandt (1972), p.34
#
Kdata<-read.table("http://web.pdx.edu/~crkl/ceR/data/klein.txt",header=T,nrows=22)
# Year: 1920 -1941 
# C: Consumption in billions of 1934 dollars.
# P: Private profits.
# I: Investment.
# W1: Private wage bill.
# W2: Government wage bill.
# G: Government nonwage spending.
# T: Indirect taxes plus net exports.
# X: Total private income before taxes, or
# X = Y + T - W2 where Y is after taxes income.
# K1: Capital stock in the begining year, or
# capital stock lagged one year. 
# K1[1942]=209.4
Kdata$P1<-c(NA,Kdata$P[2:length(Kdata$P)-1])
Kdata$X1<-c(NA,Kdata$X[2:length(Kdata$X)-1])
Kdata$W<-Kdata$W1+Kdata$W2
Kdata$K<-c(Kdata$K1[2:length(Kdata$K1)],209.4)
Kdata$Trend<-Kdata$YEAR-1931
Kdata<-subset(Kdata,Kdata$YEAR>1920)
summary(Kdata)
