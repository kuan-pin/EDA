# Example 13.2 Klein's Model I
# Simultaneous Equations Estimation
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import IV3SLS

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

# 3SLS using formula
# try different cov_type ('unadjusted', 'kernel', 'robust')
C_formula = 'C~1+P1+[P+W~G+T+W2+A+K1+X1]'
I_formula = 'I~1+P1+K1+[P~G+T+W2+A+X1]'
W_formula = 'W1~1+X1+A+[X~G+T+W2+K1+P1]'
system = {'consumption':C_formula, 'investment':I_formula, 'wage':W_formula}
klein = IV3SLS.from_formula(system,data=Kdata_)
klein_3sls = klein.fit(cov_type='unadjusted')
print(klein_3sls.summary)
klein_i3sls = klein.fit(iterate=True,cov_type='unadjusted')
print(klein_i3sls.summary)
print(klein_i3sls.cov)
print(klein_i3sls.sigma)

# 3SLS using formula with restriction
# C1~ 1 + P + P1 + (W1 + W2)
# try different cov_type ('unadjusted', 'kernel')
C1_formula = 'C~1+P1+W2+[W1+P~G+T+A+K1+X1]'
system1 = {'consumption':C1_formula, 'investment':I_formula, 'wage':W_formula}
klein1 = IV3SLS.from_formula(system1,data=Kdata_)
# restriction matrix:   1 P1  W2 W1  P  1 P1 K1  P  1 X1  A  X
cons_r1 = pd.DataFrame([[0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
klein1.add_constraints(cons_r1)
klein1_3sls = klein1.fit(cov_type='unadjusted')
print(klein1_3sls.summary)
klein1_i3sls = klein1.fit(iterate=True,cov_type='unadjusted')
print(klein1_i3sls.summary)
print(klein1_i3sls.cov)
print(klein1_i3sls.sigma)

# Other methods could be used: IVSystemGMM
