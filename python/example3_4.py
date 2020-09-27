"""
# Example 3.4
# Cobb-Douglas Production Function
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt',sep='\s+',nrows=39)
year=data['YEAR']
X=np.log(data['X'])
L=np.log(data['L1']) 
K=np.log(data['K1'])

df = pd.concat([X, L, K], keys=['X', 'L', 'K'], axis=1) 
model = smf.ols(formula='X~L + K', data=df).fit()
print(model.summary())

from statsmodels.stats.anova import anova_lm 
anova_lm(model)

# constant returns to scale: restricted least squares
# linear restrictions via parameter subsititution
modelr = smf.ols(formula='(X-K) ~ (L-K)', data=df).fit()
print(modelr.summary())
anova_lm(modelr)
anova_lm(modelr,model)

# alternatively, use restricted LS
model_constrained = smf.glm('X~L+K', data=df).fit_constrained(([0, 1, 1], 1)) 
print(model_constrained.summary())

# hypothsis testing
h1 = '(L+K=1)'
model.wald_test(h1)
model.f_test(h1)
model.t_test(h1)



