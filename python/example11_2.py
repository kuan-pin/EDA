# Example 11.2 Lagged Dependent Variable Model
# Instrumental Variable Estimation
# Using linearmodels package
import numpy as np
import pandas as pd
import statsmodels.api as sm
# pip install linearmodels
from linearmodels import IV2SLS

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",index_col='YEAR',sep='\s+',nrows=66)
y = data['Y']
c = data['C']
c1 = c.shift(1)
y1 = y.shift(1)
y2 = y.shift(2)
df = pd.DataFrame({"y":y,"c":c,"c1":c1,"y1":y1,"y2":y2})
df = df.dropna()

# ols1 = sm.OLS.from_formula('c ~ y + c1',df).fit()
model_ols = IV2SLS.from_formula('c ~ 1 + y + c1',df)
ols1 = model_ols.fit() # robust HCV
ols2 = model_ols.fit(cov_type='unadjusted')

model_iv = IV2SLS.from_formula('c ~ 1 + y + [c1 ~ y1 + y2]',df)
iv1 = model_iv.fit() # robust HCV
print(iv1.summary)
iv2 = model_iv.fit(cov_type='kernel')  # HACV
print(iv2.summary)
iv3 = model_iv.fit(cov_type='unadjusted') # Homoscedastic CV
print(iv3.summary)

# compare model results based on linearmodels
from linearmodels.iv.results import compare
res = {'OLS':ols2,'IV':iv3,'OLS-hcv':ols1,'IV-hcv':iv1,'IV-hacv':iv2}
print(compare(res))

# Tests for IV specification 
# (1) Durbin's test of exogeneity
print(iv1.durbin())      # iv1.durbin(['c1'])
# (2) Wu-Hausman test of exogeneity
print(iv1.wu_hausman())  # iv1.wu_hausman(['c1'])
# Wooldridge's regression test of exogeneity
print(iv1.wooldridge_regression)
# Wooldridge's score test of exogeneity
print(iv1.wooldridge_score)
# Wooldridge's score test of overidentification
print(iv1.wooldridge_overid)
# Sargan's test of overidentification
print(iv1.sargan)

# First Stage Diagnostics
print(iv1.first_stage)


# alternatively, using 
# from statsmodels.sandbox.regression.gmm import IV2SLS
# define xvar and ivar, then call (from_formula can not be used)
# model_iv = IV2SLS(c, xvar, ivar).fit()
