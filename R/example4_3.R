# Example 4.3
# Testing for Structural Change

setwd("C:/Course17/ceR/R")
data<-read.table("http://web.pdx.edu/~crkl/ceR/data/cjx.txt",header=T,nrows=39)

year<-data$YEAR
X<-log(data$X)
L<-log(data$L1) 
K<-log(data$K1)

break_year<-factor(year>1948)    # dummy variable

results<-lm(X~(L+K)*break_year)  # unrestricted model
summary(results)
RSSur<-sum(results$residuals^2)
DFur<-results$df.residual

results1<-lm(X~L+K)  # unrestricted model
summary(results1)
RSSr<-sum(results1$residuals^2)
DFr<-results1$df.residual

Ftest<-((RSSr-RSSur)/(DFr-DFur))/(RSSur/DFur)
Ftest
1-pf(Ftest,DFr-DFur,DFur)

# alternatively, do ANOVA
anova(results1,results)

rm(list=ls())
