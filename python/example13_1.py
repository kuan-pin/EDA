# Example 13.1 Klein's Model I
# Single Equation Estimation

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import IV2SLS, IVLIML

Kdata = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/klein.txt',sep='\s+',nrows=22)
# Year: 1920 -1941 
# C: Consumption in billions of 1934 dollars.
# P: Private profits.
# I: Investment.
# W1: Private wage bill.
# W2: Government wage bill.
# G: Government nonwage spending.
# T: Indirect taxes plus net exports.
# X: Total private income before taxes, or
# X = Y + T - W2 where Y is after taxes income.
# K1: Capital stock in the begining year, or
# capital stock lagged one year. 
# K1[1942]=209.4
P1 = Kdata.P.shift(1)
X1 = Kdata.X.shift(1)
W = Kdata.W1 + Kdata.W2
K = Kdata.K1.shift(-1)
K[21] = 209.4
A = Kdata.YEAR[1:] - 1931

Kdata_ = pd.concat([Kdata, P1, X1, W, K, A], axis=1).dropna()
Kdata_.columns = ['YEAR','C','P','W1','I','K1','X','W2','G','T','P1','X1','W','K','A']
Kdata_ = sm.add_constant(Kdata_)
'''
# OLS using formula
C_ols = sm.OLS.from_formula('C ~ P + P1 + W ', data=Kdata_).fit()
print(C_ols.summary())
I_ols = sm.OLS.from_formula('I ~ P + P1 + K1 ', data=Kdata_).fit()
print(I_ols.summary())
W_ols = sm.OLS.from_formula('W1 ~ X + X1 + A', data=Kdata_).fit()
print(W_ols.summary())
'''
# OLS using formula based on IV2SLS (for comparison with other methods)
C_ols = IV2SLS.from_formula('C ~ P + P1 + W ', data=Kdata_).fit(cov_type='unadjusted')
print(C_ols.summary)
I_ols = IV2SLS.from_formula('I ~ P + P1 + K1 ', data=Kdata_).fit(cov_type='unadjusted')
print(I_ols.summary)
W_ols = IV2SLS.from_formula('W1 ~ X + X1 + A', data=Kdata_).fit(cov_type='unadjusted')
print(W_ols.summary)

# 2SLS or IV using formula
# try different cov_type ('unadjusted', 'kernel', 'robust')
C_formula = 'C~1+P1+[P+W~G+T+W2+A+K1+X1]'
I_formula = 'I~1+P1+K1+[P~G+T+W2+A+X1]'
W_formula = 'W1~1+X1+A+[X~G+T+W2+K1+P1]'
C_2sls = IV2SLS.from_formula(C_formula, data=Kdata_).fit(cov_type='unadjusted')
print(C_2sls.summary)
I_2sls = IV2SLS.from_formula(I_formula, data=Kdata_).fit(cov_type='unadjusted')
print(I_2sls.summary)
W_2sls = IV2SLS.from_formula(W_formula, data=Kdata_).fit(cov_type='unadjusted')
print(W_2sls.summary)

# LIML using formula
C_liml = IVLIML.from_formula(C_formula, data=Kdata_).fit(cov_type='unadjusted')
print(C_liml.summary)
I_liml = IVLIML.from_formula(I_formula, data=Kdata_).fit(cov_type='unadjusted')
print(I_liml.summary)
W_liml = IVLIML.from_formula(W_formula, data=Kdata_).fit(cov_type='unadjusted')
print(W_liml.summary)

# compare single-equation model results 
from linearmodels.iv.results import compare
C_res = {'OLS':C_ols,'2SLS':C_2sls,'LIML':C_liml}
I_res = {'OLS':I_ols,'2SLS':I_2sls,'LIML':I_liml}
W_res = {'OLS':W_ols,'2SLS':W_2sls,'LIML':W_liml}
print(compare(C_res))
print(compare(I_res))
print(compare(W_res))

# alternative method
# 2SLS or IV using data matrix (exog, endog, instruments)
C_iv = IV2SLS(Kdata_.C,exog = Kdata_[['const','P1']],endog = Kdata_[['P','W']],
                instruments = Kdata_[['G','T','W2','A','K1','X1']]).fit(cov_type='unadjusted')
print(C_iv.summary)
I_iv = IV2SLS(Kdata_.I, exog = Kdata_[['const','P1', 'K1']], endog = Kdata_['P'],
                instruments = Kdata_[['G','T','W2','A','X1']]).fit(cov_type='unadjusted')
print(I_iv.summary)
W_iv = IV2SLS(Kdata_.W1, exog = Kdata_[['const','X1', 'A']], endog = Kdata_['X'],
                instruments = Kdata_[['G','T','W2','P1','K1']]).fit(cov_type='unadjusted')
print(W_iv.summary)

# Other methods could be used: IVGMM 
# alternatively,
# from statsmodels.sandbox.regression.gmm import IV2SLS
# IV = Kdata_[['G', 'T', 'W2', 'A', 'K1', 'P1', 'X1', 'const']]
# C_iv = IV2SLS(Kdata_.C, Kdata_[['P','P1','W','const']], IV).fit()
# I_iv = IV2SLS(Kdata_.I, Kdata_[['P', 'P1', 'K1','const']], IV).fit()
# W_iv = IV2SLS(Kdata_.W1, Kdata_[['X', 'X1', 'A','const']], IV).fit()
