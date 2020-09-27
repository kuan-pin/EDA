# Example 16.4 Panel Data Analysis for Investment Demand
# Deviation Approach

import numpy as np
import pandas as pd
from scipy import stats

# read 5 data files
data1 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcgm.txt", sep='\s+', nrows=20)
data2 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcch.txt", sep='\s+', nrows=20)
data3 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcge.txt", sep='\s+', nrows=20)
data4 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcwe.txt", sep='\s+', nrows=20)
data5 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcus.txt", sep='\s+', nrows=20)
data1['ID'] = 1
data2['ID'] = 2
data3['ID'] = 3
data4['ID'] = 4
data5['ID'] = 5
# set up panel data
data = pd.concat([data1,data2,data3,data4,data5])
data.set_index(['ID', 'YEAR'], inplace=True)
# switch ID and YEAR to study the time effects
# data.set_index(['YEAR', 'ID'], inplace=True)

# Pooled OLS estimator
from linearmodels import PooledOLS
pooled = PooledOLS.from_formula('I ~ 1 + F + C', data).fit()
print(pooled)

# Between estimator
from linearmodels import BetweenOLS
between = BetweenOLS.from_formula('I ~ 1 + F + C', data).fit()
print(between)

# First differences estimator (without constant term)
from linearmodels import FirstDifferenceOLS
firstdiff = FirstDifferenceOLS.from_formula('I ~ F + C', data).fit()
print(firstdiff)

from linearmodels.panel.results import compare
res1 = {'Pooled':pooled,'Between':between,'firstdiff':firstdiff}
print(compare(res1))

# Fixed effects or within estimator, constant included
from linearmodels import PanelOLS
fixed = PanelOLS.from_formula('I ~ 1 + F + C + EntityEffects', data).fit()
print(fixed)
# extract fixed effects
fixed.estimated_effects
fixed_effects = fixed.params.Intercept + fixed.estimated_effects.unstack(level=0).values[0]
print(fixed_effects)
# F test for fixed effects versus OLS
print(fixed.f_pooled)

# Random effects estimator
from linearmodels import RandomEffects
random = RandomEffects.from_formula('I ~ 1 + F + C + EntityEffects', data).fit()
print(random)
# extract fixed effects
random.estimated_effects
random_effects = random.params.Intercept + random.estimated_effects.unstack(level=0).values[0]
print(random_effects)

# compare fixed effects and random effects models
res2 = {'Pooled':pooled,'Fixed':fixed,'Random':random}
print(compare(res2))

# compare the estimates of fixed effects and random effects
effects = pd.DataFrame({'Fixed Effects':fixed_effects,'Random Effects':random_effects},
                       index=data.index.levels[0])
print(effects)

# LM test for random effects versus OLS
n = data.index.levels[0].size
T = data.index.levels[1].size
D = np.kron(np.eye(n), np.ones(T)).T
e = pooled.resids
LM = (e.dot(D).dot(D.T).dot(e) / e.dot(e) - 1) ** 2 * n * T / 2 / (T - 1)
LM_pvalue = stats.chi2(1).sf(LM)
print("LM Test: chisq = {0}, df = 1, p-value = {1}".format(LM, LM_pvalue))

# Hausman test for fixed versus random effects model
psi = fixed.cov.iloc[1:,1:] - random.cov.iloc[1:,1:]
diff = fixed.params[1:] - random.params[1:]
W = -diff.dot(np.linalg.inv(-psi)).dot(-diff)
dof = random.params.size - 1
pvalue = stats.chi2(dof).sf(W)
print("Hausman Test: chisq = {0}, df = {1}, p-value = {2}".format(W, dof, pvalue))
