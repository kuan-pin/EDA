# Example 13.4 Berndt-Wood Model Extended
# Berndt-Wood Factor Share Equations + cost function
# Elasticities Interpretation

import numpy as np
import pandas as pd
from linearmodels import SUR

bwq = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/bwq.txt',sep='\s+',nrows=25,index_col='YEAR')
bwp = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/bwp.txt',sep='\s+',nrows=25,index_col='YEAR')
bwdata = pd.concat([bwp,bwq],axis=1)
print(bwdata.describe())

PK = bwdata.PK
K = bwdata.K
PL = bwdata.PL
L = bwdata.L
PE = bwdata.PE
E = bwdata.E
PM = bwdata.PM
M = bwdata.M
tc = PK * K + PL * L + PE * E + PM * M
sk = PK * K / tc
sl = PL * L / tc
se = PE * E / tc
# sm = PM * M / tc
# sk+sl+se+sm=1
# log prices are relative to PM (linear homogeneity)
lnpk = np.log(PK / PM)
lnpl = np.log(PL / PM)
lnpe = np.log(PE / PM)

df1 = pd.DataFrame({'sk':sk,'sl':sl,'se':se,'lnpk':lnpk,'lnpl':lnpl,'lnpe':lnpe})

# formula: share equations (basicmodel)
fsk = 'sk~lnpk+lnpl+lnpe+1'
fsl = 'sl~lnpk+lnpl+lnpe+1'
fse = 'se~lnpk+lnpl+lnpe+1'

# generalization to include cost function
Q = bwdata.Q
lnc = np.log(tc / PM / Q)
lnpkpk = lnpk * lnpk / 2
lnplpl = lnpl * lnpl / 2
lnpepe = lnpe * lnpe / 2
lnpkpl = lnpk * lnpl
lnpkpe = lnpk * lnpe
lnplpe = lnpl * lnpe

df2 = pd.DataFrame({'lnc':lnc, 'lnpkpk': lnpkpk, 'lnplpl': lnplpl, 'lnpepe': lnpepe,
                      'lnpkpl': lnpkpl, 'lnpkpe': lnpkpe, 'lnplpe': lnplpe})
df = pd.concat([df1,df2],axis=1)

# formula: cost function
fc = 'lnc~lnpk+lnpl+lnpe+lnpkpk+lnpkpl+lnpkpe+lnplpl+lnplpe+lnpepe+1'

model = {'SK':fsk, 'SL':fsl, 'SE':fse, 'C':fc}
model_sur = SUR.from_formula(model,data=df)

# restriction matrix R 12x22, Q 12x1
#  1  K  L  E  1  K  L  E  1  K  L  E  1  K  L  E  KK KL KE LL LE EE
cons_r = pd.DataFrame(
[[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   
 [ 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   
 [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  
 [ 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   
 [ 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
 [ 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   
 [ 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   
 [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   
 [ 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   
 [ 0, 0, 0, 1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   
 [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
cons_q = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0])
model_sur.add_constraints(cons_r,cons_q)
bwmodel = model_sur.fit(iterate=True,cov_type='unadjusted')
print(bwmodel.summary)
print('Asymptotic Variance-Covariance Matrix of Equations')
print(bwmodel.resids.cov())

# model interpretation based on elasticity
# elasticity of i (xk) w.r.t. j (pe)
# compute own price elasticities at the means
# compute cross price elasticities at the means
bii = bwmodel.params['SK_lnpk']
bij = bwmodel.params['SK_lnpe']
si = bwmodel.fitted_values.SK.mean()
sj = bwmodel.fitted_values.SE.mean()

elast_sij = 1 + bij / si / sj
elast_pij = sj * elast_sij
elast_sii = 1 + (bii - 1) / si / si
print(elast_sij)
print(elast_pij)
