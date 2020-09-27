"""
# Example 3.3
# Multiple Regression
"""
%cd C:/Course20/ceR/python

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)
PGNP=data['PGNP']
GNP=data['GNP']/1000
POPU=data['POP']/1000
EM=data['EM']/1000
RGNP=100*GNP/PGNP

df=pd.concat([EM,RGNP,POPU])
results1=smf.ols('EM~RGNP',data=df).fit()
results2=smf.ols('EM~RGNP+POPU',data=df).fit()
print(results2.summary())

from statsmodels.stats.anova import anova_lm 
anova_lm(results2)

def reg_anova(model):
    res = pd.DataFrame({'SS': [model.ess, model.ssr, model.centered_tss],
                        'DF': [model.df_model, model.df_resid, model.nobs-1],
                        'MSS': [model.mse_model, model.mse_resid, model.mse_total],
                        'F': [model.fvalue, np.nan, np.nan],
                        'Prob>F': [model.f_pvalue, np.nan, np.nan]},
                       index=['Explained', 'Residual', 'Total'])
    return res

print(reg_anova(results2))

print('Variance-Covariance Matrix of Coefficients')
print(results2.cov_params())
print('Correlation Matrix of Coefficients')
print(results2.cov_params() / results2.bse / results2.bse[:, np.newaxis])

Predicted = pd.Series(results2.fittedvalues)
Residual = pd.Series(results2.resid)
List = pd.concat([EM, Predicted, Residual],
                 keys=['Observed', 'Predicted', 'Residual'], axis=1) 
List

import matplotlib.pyplot as plt 
resid1 = results1.resid
resid2 = results2.resid 

resid1.plot(label='resid1') 
resid2.plot(label='resid2') 
plt.legend(loc='best')

resid1.plot.density(label='resid1') 
resid2.plot.density(label='resid2') 
plt.legend(loc='best')

%who   
# clean up
%reset -f