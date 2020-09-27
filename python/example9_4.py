# Example 9.4 Multiplicative Heteroscedasticity
# Multiplicative Heteroscedasticity

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/greene.txt',sep='\s+',nrows=51)
data = data.dropna()  # take care of missing obs
print(data.describe())

spending = data['SPENDING']
income = data['INCOME']/10000

df1 = pd.concat([spending,income],keys=['spending','income'],axis=1)
model0 = smf.ols(formula='spending~income+I(income**2)',data = df1).fit()
b0 = model0.params

# log likelihood function
from numpy import sqrt, mean, log, exp
from scipy import stats, optimize

def llf1obs(b):
    h = income**b[3]
    e = (spending-b[0]-b[1]*income-b[2]*income**2)/sqrt(h)
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt(mean(e**2))))
    return (ll-0.5*log(h))

def llf2obs(b):
    h = exp(income*b[3])
    e = (spending-b[0]-b[1]*income-b[2]*income**2)/sqrt(h)
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt(mean(e**2))))
    return (ll-0.5*log(h))

# Maximum Likelihood using statsmodels' GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModel

class llf_het1(GenericLikelihoodModel):
    def loglikeobs(self, params):
        b = params
        return llf1obs(b)

class llf_het2(GenericLikelihoodModel):
    def loglikeobs(self, params):
        b = params
        return llf2obs(b)

b = list(b0) + [0]  # initial values
    
het1_model = llf_het1(spending,income)  # data input is a dummy
het1_results = het1_model.fit(b,method='newton')
print(het1_results.summary())

het2_model = llf_het2(spending,income)  # data input is a dummy
het2_results = het2_model.fit(b,method='newton')
print(het2_results.summary())
