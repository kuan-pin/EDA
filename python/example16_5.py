# Example 16.5 Panel Data Analysis for Investment Demand Function
# Seemingly Unrelated Regression Estimation

import numpy as np
import pandas as pd
from linearmodels.system import SUR

# read 5 data files
data1 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcgm.txt", sep='\s+', nrows=20, index_col='YEAR')
data2 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcch.txt", sep='\s+', nrows=20, index_col='YEAR')
data3 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcge.txt", sep='\s+', nrows=20, index_col='YEAR')
data4 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcwe.txt", sep='\s+', nrows=20, index_col='YEAR')
data5 = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/ifcus.txt", sep='\s+', nrows=20, index_col='YEAR')

data1.columns = ['I_GM', 'F_GM', 'C_GM']
data2.columns = ['I_CH', 'F_CH', 'C_CH']
data3.columns = ['I_GE', 'F_GE', 'C_GE']
data4.columns = ['I_WE', 'F_WE', 'C_WE']
data5.columns = ['I_US', 'F_US', 'C_US']

data = pd.concat([data1,data2,data3,data4,data5],axis=1)

# Perform a SUR Estimation
formulas = {'GM': 'I_GM ~ 1 + F_GM + C_GM',
            'CH': 'I_CH ~ 1 + F_CH + C_CH',
            'GE': 'I_GE ~ 1 + F_GE + C_GE',
            'WE': 'I_WE ~ 1 + F_WE + C_WE',
            'US': 'I_US ~ 1 + F_US + C_US'}
model = SUR.from_formula(formulas, data)
model_sur = model.fit(iterate=True, cov_type='unadjusted')
print(model_sur.summary)
print('Asymptotic Variance-Covariance Matrix of Equations')
print(model_sur.resids.cov())

# restriction matrix R: 8x15, Q: 8x1
#                       1  F  C  1  F  C  1  F  C  1  F  C  1  F  C
cons_r = pd.DataFrame([[0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
                       [0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]])
cons_q = pd.Series([0,0,0,0,0,0,0,0])
model.add_constraints(cons_r,cons_q)
modelr_sur = model.fit(iterate=True,cov_type='unadjusted')
print(modelr_sur.summary)
print('Asymptotic Variance-Covariance Matrix of Equations')
print(modelr_sur.resids.cov())
