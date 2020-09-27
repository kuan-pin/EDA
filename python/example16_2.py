# Example 16.2 One-Way Panel Data Analysis, Deviation Approach
# Production of Airline Services: C = f(Q,PF,LF)
# Panel data: 6 airline companies, 15 years (1970-1984)
# Fixed effects and random effects models

import numpy as np
import pandas as pd
from scipy import stats
# set up panel data
pdata = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/airline.txt", sep='\s+', nrows=90, index_col=['I','T'])
pdata.describe()
# alternatively
# data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/airline.txt", sep='\s+', nrows=90)
# Set data as panel data
# pdata = data.set_index(['I', 'T'], inplace=True)

# variable transfermation
cs = np.log(pdata.C)
qs = np.log(pdata.Q)
pfs = np.log(pdata.PF)
lfs = pdata.LF # load factor, not logged
df = pd.DataFrame({"cs": cs, "qs": qs, "pfs": pfs, "lfs": lfs})
# Descriptive statistics
df.describe()

# Pooled OLS estimator
from linearmodels import PooledOLS
pooled = PooledOLS.from_formula('cs ~ 1 + qs + pfs + lfs', df).fit()
print(pooled)

# Between estimator
from linearmodels import BetweenOLS
between = BetweenOLS.from_formula('cs ~ 1 + qs + pfs + lfs', df).fit()
print(between)

# First differences estimator (without constant term)
from linearmodels import FirstDifferenceOLS
firstdiff = FirstDifferenceOLS.from_formula('cs ~ qs + pfs + lfs', df).fit()
print(firstdiff)

from linearmodels.panel.results import compare
res1 = {'Pooled':pooled,'Between':between,'firstdiff':firstdiff}
print(compare(res1))


# Fixed effects or within estimator
# with constant inclued or not, will have the same results
# with constant term surpressed
from linearmodels import PanelOLS
fixed = PanelOLS.from_formula('cs ~ qs + pfs + lfs + EntityEffects', df).fit()
print(fixed)
# extract fixed effects
fixed.estimated_effects
fixed_effects = fixed.estimated_effects.unstack(level=0).values[0]
print(fixed_effects)
# F test for fixed effects versus OLS
print(fixed.f_pooled)

# with constant term included 
fixed1 = PanelOLS.from_formula('cs ~ 1 + qs + pfs + lfs + EntityEffects', df).fit()
print(fixed1)
# extract fixed effects
fixed1.estimated_effects
fixed1_effects = fixed1.params.Intercept + fixed1.estimated_effects.unstack(level=0).values[0]
print(fixed1_effects)
# F test for fixed effects versus OLS
print(fixed1.f_pooled)

# Random effects estimator, constant term must be included
# should not have EntityEffects or TimeEffects in the formula
from linearmodels import RandomEffects
random = RandomEffects.from_formula('cs ~ 1 + qs + pfs + lfs', df).fit()
print(random)
# extract fixed effects
random.estimated_effects
random_effects = random.params.Intercept + random.estimated_effects.unstack(level=0).values[0]
print(random_effects)
print(random.variance_decomposition)

# compare fixed effects and random effects models
res2 = {'Pooled':pooled,'Fixed+1':fixed1,'Fixed':fixed,'Random':random}
print(compare(res2))

effects = pd.DataFrame({'Fixed Effects':fixed_effects,'Random Effects':random_effects},
                       index=pdata.index.levels[0])
print(effects)

# LM test for random effects versus OLS
n = pdata.index.levels[0].size
T = pdata.index.levels[1].size
D = np.kron(np.eye(n), np.ones(T)).T
e = pooled.resids
LM = (e.dot(D).dot(D.T).dot(e) / e.dot(e) - 1) ** 2 * n * T / 2 / (T - 1)
LM_pvalue = stats.chi2(1).sf(LM)
print("LM Test: chisq = {0}, df = 1, p-value = {1}".format(LM, LM_pvalue))

# Hausman test for fixed versus random effects model
# null hypothesis: random effects model
psi = fixed.cov - random.cov.iloc[1:,1:]
diff = fixed.params - random.params[1:]
# psi = fixed1.cov.iloc[1:,1:] - random.cov.iloc[1:,1:]
# diff = fixed1.params[1:] - random.params[1:]
W = diff.dot(np.linalg.inv(psi)).dot(diff)
dof = random.params.size -1
pvalue = stats.chi2(dof).sf(W)
print("Hausman Test: chisq = {0}, df = {1}, p-value = {2}".format(W, dof, pvalue))

# alternative hausman test based on random effects model
# include group means in the random effects model
# test the significance of group mean coefficients
df2 = df
df2['qsm'] = np.kron(df2.qs.mean(level=0), np.ones(T))
df2['pfsm'] = np.kron(df2.pfs.mean(level=0), np.ones(T))
df2['lfsm'] = np.kron(df2.lfs.mean(level=0), np.ones(T))
random1 = RandomEffects.from_formula('cs ~ 1 + qs + pfs + lfs + qsm + pfsm + lfsm', df2).fit(cov_type='clustered',cluster_entity=True)
print(random1)
# hypothsis testing of 'qsm=pfsm=lfsm=0'
print(random1.wald_test(formula='qsm=pfsm=lfsm=0'))

# panel robust hetero cov
fixed_robust = PanelOLS.from_formula('cs ~ 1 + qs + pfs + lfs + EntityEffects', df).fit(cov_type='clustered',cluster_entity=True)
print(fixed_robust)
random_robust = RandomEffects.from_formula('cs ~ 1 + qs + pfs + lfs', df).fit(cov_type='clustered',cluster_entity=True)
print(random_robust)

# compare fixed effects and random effects models
res3 = {'Fixed (Panel-Robust)':fixed_robust,'Random (Panel-Robust)':random_robust}
print(compare(res3))
