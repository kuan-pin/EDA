# Example 16.3 Two-Way Panel Data Analysis, Dummy Variable
# Cost of Production for Airline Services II

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

# when using formula, try to aviod using variables named 'I' or 'C'
model = sm.OLS.from_formula('cs ~ qs + pfs + lfs + D + T', df).fit()
print(model.summary())

model1 = sm.OLS.from_formula('cs ~ qs + pfs + lfs + D', df).fit()
print(model1.summary())

model2 = sm.OLS.from_formula('cs ~ qs + pfs + lfs + T', df).fit()
print(model2.summary())

model3 = sm.OLS.from_formula('cs ~ qs + pfs + lfs', df).fit()
print(model3.summary())

print(anova_lm(model3,model1))  # individual one-way effect
print(anova_lm(model3,model2))  # time one-way effect
print(anova_lm(model3,model))   # two-way effect
print(anova_lm(model2,model))
print(anova_lm(model1,model))

# alternatively, using panel data
pdata = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/airline.txt", sep='\s+', nrows=90, index_col=['I','T'])
cs = np.log(pdata['C'])
qs = np.log(pdata['Q'])
pfs = np.log(pdata['PF'])
lfs = pdata['LF'] # load factor, not logged
pdf = pd.DataFrame({'cs': cs, 'qs': qs, 'pfs': pfs, 'lfs': lfs})

from linearmodels import PanelOLS
pmodel = PanelOLS.from_formula('cs ~ 1 + qs + pfs + lfs + EntityEffects + TimeEffects', pdf).fit()
print(pmodel.summary)
print(pmodel.f_pooled)
two_way_effects=pmodel.params.Intercept + pmodel.estimated_effects
print(two_way_effects.unstack())

pmodel1 = PanelOLS.from_formula('cs ~ 1 + qs + pfs + lfs + EntityEffects', pdf).fit()
print(pmodel1.summary)
print(pmodel1.f_pooled)
entity_effects = pmodel1.params.Intercept + pmodel1.estimated_effects
print(entity_effects.unstack(level='I').values[0])

pmodel2 = PanelOLS.from_formula('cs ~ 1 + qs + pfs + lfs + TimeEffects', pdf).fit()
print(pmodel2.summary)
print(pmodel2.f_pooled)
time_effects=pmodel2.params.Intercept + pmodel2.estimated_effects
print(time_effects.unstack(level='T').values[0])
