# Example 11.4 Autoregressive Distributed Lag Model
# Almon Lag Model Once More, using data from 1953Q1 - 1961Q4
# Almon Lag (Lags=7, Order=4 End=2): tested and estimated

import pandas as pd
import numpy as np
import statsmodels.api as sm

almon = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/almon.txt",sep='\s+',nrows=36)
cexp = almon['CEXP']
capp = almon['CAPP']

X = pd.concat([capp,capp.shift(1),capp.shift(2),capp.shift(3),capp.shift(4),
              capp.shift(5),capp.shift(6),capp.shift(7)], axis=1)
X.columns = ['CAPP','CAPP1','CAPP2','CAPP3','CAPP4','CAPP5','CAPP6','CAPP7'] 
Y = pd.concat([cexp.shift(1),cexp.shift(2)],axis=1)
Y.columns = ['CEXP1','CEXP2']

xvar = pd.concat([Y,X],axis=1)
xvar = sm.add_constant(xvar)
xvar = xvar.dropna()
yvar = cexp.iloc[7:]

# unrestricted PDL 7-lags model using OLS
pdl1 = sm.OLS(yvar, xvar).fit()
print(pdl1.summary())

def PDL(q,p):
    Hmat = np.empty((q,p))
    for i in range(q):
        for j in range(p):
            Hmat[i,j]=(i+1)**(j+1)
    Hmat = np.concatenate(([np.zeros(p)],Hmat))
    Hmat = np.concatenate((np.ones(q+1).reshape(-1,1),Hmat),axis=1)
    return(Hmat)

# polynomial lags: restricted, lags=7 order=4
H = PDL(7,4)
Z = X @ H
Z.columns = ['Z0','Z1','Z2','Z3','Z4']
zvar = pd.concat([Y,Z],axis=1)
zvar = sm.add_constant(zvar)
zvar = zvar.dropna()
yvar = cexp.iloc[7:]

pdl2 = sm.OLS(yvar,zvar).fit()
print(pdl2.summary())

# hypothsis testing: check for end-point restrictions
h1 = '(Z0-Z1+Z2-Z3+Z4=0)'                # left-end restriction
h2 = '(Z0+8*Z1+64*Z2+512*Z3+4096*Z4=0)'  # right-end restriction
print(pdl2.wald_test(h1))
print(pdl2.wald_test(h2))
print(pdl2.wald_test([h1,h2]))

# PDL(7,4) with both end-point restrictions
from statsmodels.api import GLM
cons_r = [[0,0,0,1,-1,1,-1,1],
          [0,0,0,1,8,64,512,4096]]
cons_q = [0,0]
pdl3 = sm.GLM(yvar,zvar).fit_constrained((cons_r,cons_q))
print(pdl3.summary())

a_coef = H@pdl3.params[3:]
v = np.array(pdl3.cov_params())
a_se = np.sqrt(np.diag(H @ v[3:,3:] @ H.T))
a_z = a_coef/a_se
res = pd.DataFrame({'coef':a_coef,'std err':a_se,'z':a_z},
                   index=X.columns)
print(res)
