# Example 13.3 Berndt-Wood Model
# Berndt-Wood Factor Share Equations
# Basic Model: 3 Shares in relative price with restrictions
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
# formula: share equations
fsk = 'sk~1+lnpk+lnpl+lnpe'
fsl = 'sl~1+lnpk+lnpl+lnpe'
fse = 'se~1+lnpk+lnpl+lnpe'

df1 = pd.DataFrame({'sk':sk,'sl':sl,'se':se,'lnpk':lnpk,'lnpl':lnpl,'lnpe':lnpe})
model1 = {'SK':fsk, 'SL':fsl, 'SE':fse}
model1_sur = SUR.from_formula(model1,data=df1)
# restriction matrix R: 3x12, Q: 3x1
cons_r1 = pd.DataFrame([[0,0,1,0,0,-1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,-1,0,0],
                        [0,0,0,0,0,0,0,1,0,0,-1,0]])
cons_q1 = pd.Series([0,0,0])
model1_sur.add_constraints(cons_r1,cons_q1)
bwmodel1 = model1_sur.fit(iterate=True,cov_type='unadjusted')
print(bwmodel1.summary)
print('Asymptotic Variance-Covariance Matrix of Equations')
print(bwmodel1.resids.cov())

# model interpretation based on elasticity
# elasticity of i (xk) w.r.t. j (pe)
# compute own price elasticities at the means
# compute cross price elasticities at the means
bii = bwmodel1.params['SK_lnpk']
bij = bwmodel1.params['SK_lnpe']
si = bwmodel1.fitted_values.SK.mean()
sj = bwmodel1.fitted_values.SE.mean()

elast_sij = 1 + bij / si / sj
elast_pij = sj * elast_sij
elast_sii = 1 + (bii - 1) / si / si
print(elast_sij)
print(elast_pij)
