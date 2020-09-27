# Example 12.3 GMM Estimation of U.S. Consumption Function
# See also Example 11.2, using linearmodels package

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS,IVGMM, IVGMMCUE

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",index_col='YEAR',sep='\s+',nrows=66)
y = data['Y']
c = data['C']
c1 = c.shift(1)
y1 = y.shift(1)
y2 = y.shift(2)
df = pd.DataFrame({"c":c,"c1":c1,"y":y,"y1":y1,"y2":y2})
df = df.dropna()

# model_ols = sm.OLS.from_formula('c ~ y + c1',df)
model_ols = IV2SLS.from_formula('c ~ 1 + y + c1',df)
ols1 = model_ols.fit(cov_type='unadjusted')
ols2 = model_ols.fit()  # default cov_type='robust'
print(ols2.summary)

model_iv = IV2SLS.from_formula('c ~ 1 + y + [c1 ~ y1 + y2]',df)
iv1 = model_iv.fit(cov_type='unadjusted') # try HCV-robust
print(iv1.summary)
iv2 = model_iv.fit() # default cov_type='robust'
print(iv2.summary)
# sargan test of overidentifying restrictions
print(iv2.sargan)

model_gmm = IVGMM.from_formula('c ~ 1 + y + [c1 ~ y1 + y2]',df)
gmm1 = model_gmm.fit() # 2-step, weight_type='robust'
print(gmm1.summary)
gmm2 = model_gmm.fit(iter_limit=50)
print(gmm2.summary)
# initial_weight=np.eye(3) may be used to set the weight matrix in the first iteration
model_gmmcue = IVGMMCUE.from_formula('c ~ 1 + y + [c1 ~ y1 + y2]',df)
gmm3 = model_gmmcue.fit() # weight_type='robust'
print(gmm3.summary)

# J-test of overidentifying restrictions: E(moment conditions) = 0
print(gmm3.j_stat)

# compare model results based on linearmodels
from linearmodels.iv.results import compare
res = {'OLS':ols2,'IV':iv2,'GMM-2steps':gmm1,'GMM-Iterative':gmm2,'GMM-CUE':gmm3}
print(compare(res))

# alternatively, using 
# from statsmodels.sandbox.regression.gmm import LinearIVGMM
# define xvar and ivar, then call (from_formula can not be used)
# model = LinearIVGMM(c, xvar, ivar).fit()
# see also IV2SLS, IVGMM
from statsmodels.sandbox.regression.gmm import LinearIVGMM
df = sm.add_constant(df)
yvar = df['c']
xvar = df[['const','y','c1']]
zvar = df[['const','y','y1','y2']]
model = LinearIVGMM(yvar,xvar,zvar)
linear_gmm1 = model.fit()
print(linear_gmm1.summary())  # gmm2
linear_gmm2 = model.fit(maxiter=50) # gmm2
print(linear_gmm2.summary())
linear_gmm3 = model.fit(maxiter='cue')
print(linear_gmm3.summary())
