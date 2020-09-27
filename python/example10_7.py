# Example 10.7 Maximum Likelihood Estimation
# AR(1), MA(1), ARMA(1,1)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', nrows=39)

X = np.log(data.X)
L = np.log(data.L1)
K = np.log(data.K1)

Z = pd.concat([L,K],axis=1)
Z = sm.add_constant(Z)

ols1 = sm.OLS(X,Z).fit()
print(ols1.summary())

# define log-likelihood function of AR(1) error structure
def ar1_co(y,X,b):
    rho = b[-1]
    beta = b[:-1]
    e = y-X@beta
    u = e[1:]-rho*e[:-1]
    s = np.sqrt((u**2).mean())
    llf = stats.norm.logpdf(u, loc=0, scale=s)
    return(llf)
    
# log-likelihood function for AR(1) error structure
def ar1(y,X,b):
    rho = b[-1]
    beta = b[:-1]
    e = y-X@beta
    u = e[1:]-rho*e[:-1]
    u = np.append(np.sqrt(1-rho**2)*e[0],u)
    s = np.sqrt((u**2).mean())
    llf = stats.norm.logpdf(u, loc=0, scale=s)
    llf[0] = llf[0]+np.log(np.sqrt(1-rho**2)) 
    # add log-jacobian of the first obs.
    return(llf)

# log-likelihood function for MA(1) error structure
def ma1(y,X,b):
    theta = b[-1]
    beta = b[:-1]
    e = y-X@beta
    u = e
    for i in range(1,u.shape[0]):
        u[i] = e[i] + theta*u[i-1]
    s = np.sqrt((u**2).mean())
    llf = stats.norm.logpdf(u, loc=0, scale=s)
    return(llf)  
    
# log-likelihood function for ARMA(1,1) error structure
def arma1(y,X,b):
    rho = b[-2]
    theta = b[-1]
    beta = b[:-2]
    e = y-X@beta
    u = e[1:]-rho*e[:-1]
    u = np.append(np.sqrt(1-rho**2)*e[0],u)
    v = u
    for i in range(1,v.shape[0]):
        v[i] = u[i] + theta*v[i-1]
    s = np.sqrt((v**2).mean())
    llf = stats.norm.logpdf(v, loc=0, scale=s)
    llf[0] = llf[0]+np.log(np.sqrt(1-rho**2)) 
    # add log-jacobian of the first obs.
    return(llf)

# model estimation    
# using statsmodels' GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModel

# initial values
b0 = list(ols1.params)+[0.5]

# AR(1)
class llf_ar1(GenericLikelihoodModel):
    def loglikeobs(self, params):
        y = self.endog
        X = self.exog
        b = params
        return(ar1(y,X,b))

ar1_model = llf_ar1(X,Z)
ar1_results = ar1_model.fit(b0,method='newton')
print(ar1_results.summary())

# MA(1)
class llf_ma1(GenericLikelihoodModel):
    def loglikeobs(self, params):
        y = self.endog
        X = self.exog
        b = params
        return(ma1(y,X,b))

ma1_model = llf_ma1(X,Z)
ma1_results = ma1_model.fit(b0,method='bfgs')
print(ma1_results.summary())

b0 = list(ols1.params)+[0.5,0.5]
# ARMA(1,1)
class llf_arma1(GenericLikelihoodModel):
    def loglikeobs(self, params):
        y = self.endog
        X = self.exog
        b = params
        return(arma1(y,X,b))

arma1_model = llf_arma1(X,Z)
arma1_results = arma1_model.fit(b0,method='bfgs')
print(arma1_results.summary())

# compare with TSA.ARIMA
# MA parameter is coded differently, the results are not the same
from statsmodels.tsa.arima_model import ARMA
tsa_ar1 = ARMA(X,exog=Z,order=(1,0)).fit(trend='nc',method='mle')
print(tsa_ar1.summary())
tsa_ma1 = ARMA(X,exog=Z,order=(0,1)).fit(trend='nc',method='mle')
print(tsa_ma1.summary())
tsa_arma1 = ARMA(X,exog=Z,order=(1,1)).fit(trend='nc',method='mle')
print(tsa_arma1.summary())
