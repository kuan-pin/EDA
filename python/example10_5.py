# Example 10.5 Higher Order Autocorrelation

import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', nrows=39)

X = np.log(data.X)
L = np.log(data.L1)
K = np.log(data.K1)
Z = pd.concat([L,K],axis=1)
Z = sm.add_constant(Z)
model_ols = sm.OLS(X,Z).fit()

def ols_ar(model,rho):
    x = model.model.exog
    y = model.model.endog
    p = rho.shape[0]
    ystar = y; xstar = x
    for i in range(1,p+1):
        ystar = ystar[1:]-rho[i-1]*y[:-i]
        xstar = xstar[1:,]-rho[i-1]*x[:-i,]
    model_ar = sm.OLS(ystar,xstar).fit()
    return(model_ar)

def OLSAR(model,order=1):
    x = model.model.exog
    y = model.model.endog
    e = (y - x @ model.params).reshape(-1,1)
    p = order
    e1 = e
    for i in range(1,p+1):
        e1 = np.hstack((e1[1:,],e[:-i]))
    rho0 = sm.OLS(e1[:,0],e1[:,1:]).fit().params
    print('Rho = ', rho0)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1 = ols_ar(model,rho0)
        e = (y -x @ model1.params).reshape(-1,1)
        e1 = e
        for i in range(1,p+1):
            e1 = np.hstack((e1[1:,],e[:-i]))
        rho1 = sm.OLS(e1[:,0],e1[:,1:]).fit().params
        # rho1 = sm.OLS(e0,e1).fit().params
        rdiff = np.sqrt(np.sum((rho1-rho0)**2))
        rho0 = rho1
        print('Rho = ', rho0)
    return(ols_ar(model,rho0))

model_ar = OLSAR(model_ols,3)
model_ar.summary()

# check for residual autocorrelation of the final model
e = model_ar.resid
lags = 5
bptest = sm.stats.diagnostic.acorr_ljungbox(e, lags=lags, boxpierce=True, return_df=True)
print(bptest)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(e,zero=False)
plot_pacf(e,zero=False)

# alternatively, using statsmodels' GLSAR
# estimate AR(1), AR(2), AR(3), 
# results may not be the same as OLSAR, need to check
for i in range(3):
    rho = i+1
    model_gls = sm.GLSAR(X,Z,rho)
    results = model_gls.iterative_fit(maxiter=50)
    print ('Iterations used = %d Converged %s' % (results.iter, results.converged) )
    print ('Rho =  ', model_gls.rho)
    print(results.summary())

# alternatively, ARIMA model may be used
# results may not be the same as OLSAR and GLSAR, need to check
from statsmodels.tsa.arima_model import ARMA
for i in range(3):
    rho = i+1
    armodel = ARMA(X,exog=Z,order=(rho,0)).fit(trend='nc')
    print(armodel.summary())
