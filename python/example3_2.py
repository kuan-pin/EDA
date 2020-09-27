"""
# Example 3.2
# Residual Analysis
"""
%cd C:/Course20/ceR/python

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf 

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)
PGNP=data['PGNP']
GNP=data['GNP']/1000
EM=data['EM']/1000
RGNP=100*GNP/PGNP

df1=pd.concat([EM,RGNP],axis=1)
model1=smf.ols('EM~RGNP',data=df1)
results1=model1.fit()
print(results1.summary())

from statsmodels.stats.anova import anova_lm 
anova_lm(results1)

EM_fitted = pd.Series(results1.fittedvalues)
resid1 = pd.Series(results1.resid)
List = pd.concat([EM, EM_fitted, resid1], 
                 keys=['Observed', 'Predicted', 'Residual'], axis=1) 
List

import matplotlib.pyplot as plt
%matplotlib inline

resid1.plot(ylim=(-1, 1.25))

resid1.plot.hist(density=True)
resid1.plot.density()

%who   
# clean up
%reset -f
