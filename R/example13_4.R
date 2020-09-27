# Example 13.4 Berndt-Wood Model Extended
# Berndt-Wood Factor Share Equations + cost function
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

# generalization to include cost function
c<-log((tc/PM)/Q)
lnpkpk<-0.5*lnpk*lnpk
lnplpl<-0.5*lnpl*lnpl
lnpepe<-0.5*lnpe*lnpe
lnpkpl<-lnpk*lnpl
lnpkpe<-lnpk*lnpe
lnplpe<-lnpl*lnpe
# formula: cost function
fc<-c~lnpk+lnpl+lnpe+lnpkpk+lnplpl+lnpepe+lnpkpl+lnpkpe+lnplpe
# model: including cost function
bwmodel2<-list(C=fc,SK=fsk,SL=fsl,SE=fse)
# constraints associated with cost function
res4<-"C_lnpk-SK_(Intercept)=0"
res5<-"C_lnpl-SL_(Intercept)=0"
res6<-"C_lnpe-SE_(Intercept)=0"
res7<-"C_lnpkpk-SK_lnpk=0"
res8<-"C_lnplpl-SL_lnpl=0"
res9<-"C_lnpepe-SE_lnpe=0"
res10<-"C_lnpkpl-SK_lnpl=0"
res11<-"C_lnpkpe-SK_lnpe=0"
res12<-"C_lnplpe-SL_lnpe=0"

# Iterative SUR
out2<-systemfit(bwmodel2,method="SUR",data=bwdata,maxiter=100,
                restrict.matrix=c(res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12))
summary(out2)

# model interpretation based on elasticity
# elasticity of i (xk) w.r.t. j (pe)
# compute own price elasticities at the means
# compute cross price elasticities at the means
bij<-out2$coefficients["SK_lnpe"]
si<-as.matrix(fitted.values(out2)["SK"])
sj<-as.matrix(fitted.values(out2)["SE"])

elast_sij<-1+bij/(mean(si)*mean(sj))
elast_pij<-mean(sj)*elast_sij
names(elast_sij)<-"elast_sij"
names(elast_pij)<-"elast_pij"
c(elast_sij,elast_pij)
rm(list = ls())
