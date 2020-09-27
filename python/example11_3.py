# Example 11.3 Polynomial Distributed Lag Model
# Almon Lag Model Revisited
# Use data in the original paper: 1953Q1-1961Q4
# Almon Lag (Lags=7, Order=4)
# Almon Lag (Lags=7, Order=4 End=2)+Seasonality: tested and estimated

import pandas as pd
import numpy as np
import statsmodels.api as sm
# from statsmodels.api import GLM

almon = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/almon.txt",sep='\s+',nrows=36)

cexp = almon['CEXP']
capp = almon['CAPP']
qt = pd.get_dummies(almon['YEARQT']%100, prefix='Q') 
df = pd.concat([cexp,capp,qt],axis = 1)
df.head()

ols0 = sm.OLS.from_formula('cexp~capp+qt-1',data=df).fit()
print(ols0.summary())

X = pd.concat([capp,capp.shift(1),capp.shift(2),capp.shift(3),capp.shift(4),
              capp.shift(5),capp.shift(6),capp.shift(7)], axis=1)
X.columns = ['CAPP','CAPP1','CAPP2','CAPP3','CAPP4','CAPP5','CAPP6','CAPP7'] 

xvar = pd.concat([X,qt],axis=1)
xvar = xvar.dropna()
yvar = cexp.iloc[7:]

# unrestricted PDL (OLS)
ols1 = sm.OLS(yvar,xvar).fit()
print(ols1.summary())

from statsmodels.api import GLM
pdl1 = sm.GLM(yvar,xvar).fit_constrained(([0,0,0,0,0,0,0,0,1,1,1,1],0))
print(pdl1.summary())

# PDL transformation matrix
# q = lags, p = orders, p<q
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
zvar = pd.concat([Z,qt],axis=1)
zvar = zvar.dropna()
yvar = cexp.iloc[7:]

ols2 = sm.OLS(yvar,zvar).fit()
print(ols2.summary())

pdl2 = sm.GLM(yvar,zvar).fit_constrained(([0,0,0,0,0,1,1,1,1],0))
print(pdl2.summary())

# hypothsis testing: check for end-point restrictions
h1 = '(Z0-Z1+Z2-Z3+Z4=0)'                # left-end restriction
h2 = '(Z0+8*Z1+64*Z2+512*Z3+4096*Z4=0)'  # right-end restriction
print(pdl2.wald_test(h1))
print(pdl2.wald_test(h2))
print(pdl2.wald_test([h1,h2]))

# PDL(7,4) with both end-point restrictions
cons_r = [[0,0,0,0,0,1,1,1,1],
          [1,-1,1,-1,1,0,0,0,0],
          [1,8,64,512,4096,0,0,0,0]]
cons_q = [0,0,0]
pdl3 = sm.GLM(yvar,zvar).fit_constrained((cons_r,cons_q))
print(pdl3.summary())

a_coef = H@pdl3.params[0:5]
v = np.array(pdl3.cov_params())
a_se = np.sqrt(np.diag(H @ v[0:5,0:5] @ H.T))
a_z = a_coef/a_se
res = pd.DataFrame({'coef':a_coef,'std err':a_se,'z':a_z},
                   index=X.columns)
print(res)
