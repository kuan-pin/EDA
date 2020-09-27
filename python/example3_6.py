"""
# Example 3.6
# Residual Diagnostics
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt',sep='\s+',nrows=39)

year=data['YEAR']
X=np.log(data['X'])
L=np.log(data['L1']) 
K=np.log(data['K1'])

df = pd.concat([X, L, K], keys=['X', 'L', 'K'], axis=1) 
model = smf.ols(formula='X ~ L + K', data=df).fit()
print(model.summary())

JBtest = stats.jarque_bera(model.resid)
JBtest[0]
JBtest[1] # p-value = 1-stats.chi2.cdf(JB[0], 2) 

# check for residual normality
import matplotlib.pyplot as plt
model.resid.plot.density()
stats.probplot(model.resid, dist='norm', plot=plt) 

# residual diagnostics
from statsmodels.stats.outliers_influence import OLSInfluence
res_diag = OLSInfluence(model).summary_frame() 
res_diag[['hat_diag', 'standard_resid', 'student_resid', 'dffits']]
 