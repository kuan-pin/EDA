# Example 13.3 Berndt-Wood Model
# Berndt-Wood Factor Share Equations
# Basic Model: 3 Shares in relative price with restrictions
# Elasticities Interpretation

bwq<-read.table("http://web.pdx.edu/~crkl/ceR/data/bwq.txt",header=T,nrows=25)
bwp<-read.table("http://web.pdx.edu/~crkl/ceR/data/bwp.txt",header=T,nrows=25)
bwdata<-merge(bwp,bwq)
summary(bwdata)
rm(bwp,bwq)

attach(bwdata)
tc<-PK*K+PL*L+PE*E+PM*M
sk<-PK*K/tc
sl<-PL*L/tc
se<-PE*E/tc
sm<-PM*M/tc
# sk+sl+se+sm=1
# log prices are relative to PM (linear homogeneity)
lnpk<-log(PK/PM)
lnpl<-log(PL/PM)
lnpe<-log(PE/PM)
detach(bwdata)

# formula: share equations
fsk<-sk~lnpk+lnpl+lnpe
fsl<-sl~lnpk+lnpl+lnpe
fse<-se~lnpk+lnpl+lnpe
bwmodel<-list(SK=fsk,SL=fsl,SE=fse)

# setup linear constraints: symmetry
res1<-"SK_lnpl-SL_lnpk=0"
res2<-"SK_lnpe-SE_lnpk=0"
res3<-"SL_lnpe-SE_lnpl=0"

library(systemfit)
# Iterative SUR
out1<-systemfit(bwmodel,method="SUR",data=bwdata,maxiter=100,
                restrict.matrix=c(res1,res2,res3))
summary(out1)

# model interpretation based on elasticity
# elasticity of i (xk) w.r.t. j (pe)
# compute own price elasticities at the means
# compute cross price elasticities at the means
bij<-out1$coefficients["SK_lnpe"]
si<-as.matrix(fitted.values(out1)["SK"])
sj<-as.matrix(fitted.values(out1)["SE"])

elast_sij<-1+bij/(mean(si)*mean(sj))
elast_pij<-mean(sj)*elast_sij
names(elast_sij)<-"elast_sij"
names(elast_pij)<-"elast_pij"
c(elast_sij,elast_pij)

rm(list = ls())
