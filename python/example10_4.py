# Example 10.4 Hildreth-Lu Grid Search Procedure

import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', nrows=39)

X = np.log(data.X)
L = np.log(data.L1)
K = np.log(data.K1)
Z = pd.concat([L,K],axis=1)
Z = sm.add_constant(Z)
model_ols = sm.OLS(X,Z).fit()

# cochrane-orcutt / prais-winsten with given AR(1) rho, 
# derived from ols model, default to cochrane-orcutt (drop=True)
def ols_ar1(model,rho,drop1=True):
    x = model.model.exog
    y = model.model.endog
    ystar = y[1:]-rho*y[:-1]
    xstar = x[1:,]-rho*x[:-1,]
    if drop1 == False:
        ystar = np.append(np.sqrt(1-rho**2)*y[0],ystar)
        xstar = np.append([np.sqrt(1-rho**2)*x[0,]],xstar,axis=0)
    model_ar1 = sm.OLS(ystar,xstar).fit()
    return(model_ar1)

# hildreth-lu grid search procedure
def OLSAR1_hl(model,drop1=True):
    r0 = 0; s0 = 1
    while s0 > 1.0e-5:
        rho = np.arange(r0-0.9*s0, r0+0.9*s0, s0/10)
        SSR = np.ones(np.shape(rho))
        j = 0
        for i in rho:
            model1 = ols_ar1(model,i,drop1)
            SSR[j] = model1.ssr
            j = j+1
        tab = pd.DataFrame({'rho': rho, 'SSR': SSR})
        r0 = tab['rho'][tab['SSR']==tab['SSR'].min()].values
        s0 = s0/10
        print('Rho = ', r0)
    model1 = ols_ar1(model,i,drop1)
    return(model1)
    
# AR(1) based on cochrane-orcutt grid search procedure   
ar1_hlco = OLSAR1_hl(model_ols)
# ar1_co = OLSAR1(model_ols,drop1=True)
print(ar1_hlco.summary())

# AR(1) based on prais-winsten grid search procedure
ar1_hlpw = OLSAR1_hl(model_ols,drop1=False)
print(ar1_hlpw.summary())
# the results are based on transformed model
