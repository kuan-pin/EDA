# Example 16.1 One-Way Panel Data Analysis, Dummy Variable
# Cost of Production for Airline Services I

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/airline.txt", sep='\s+', nrows=90)
D = data['I'].astype('category')
T = data['T'].astype('category')
cs = np.log(data['C'])
qs = np.log(data['Q'])
pfs = np.log(data['PF'])
lfs = data['LF'] # load factor, not logged
df = pd.DataFrame({'D': D, 'T': T, "cs": cs, "qs": qs, "pfs": pfs, "lfs": lfs})

# individual effects model
model1 = sm.OLS.from_formula('cs ~ qs + pfs + lfs + D', df).fit()
print(model1.summary())
print(anova_lm(model1,typ=2))

# time effects model
model2 = sm.OLS.from_formula('cs ~ qs + pfs + lfs + T', df).fit()
print(model2.summary())
print(anova_lm(model2,typ=2))

# pooled model
model = sm.OLS.from_formula('cs ~ qs + pfs + lfs', df).fit()
print(model.summary())

print(anova_lm(model,model1))
print(anova_lm(model,model2))
