# Example 4.2
# Dummy Variable Trap

import numpy as np 
import pandas as pd 
from scipy import stats 
import statsmodels.api as sm

data_almon = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/almon.txt',sep='\s+',nrows=60)
# seasoanl factor variable
'''
model1 = sm.OLS.from_formula('CEXP~CAPP+C(YEARQT%10)',data=data_almon).fit() 
print(model1.summary())
'''
QT = pd.get_dummies(data_almon['YEARQT'] % 10, prefix='Q') 

data = pd.concat([data_almon,QT],axis = 1)
data.head()

Y = data['CEXP']
X1 = data[['CAPP','Q_1','Q_2','Q_3','Q_4']]

model1 = sm.OLS(Y, X1).fit() 
print(model1.summary())

X2 = data[['CAPP','Q_1','Q_2','Q_3']]
model2 = sm.OLS(Y, sm.add_constant(X2)).fit() 
print(model2.summary())

model3 = sm.OLS(Y, sm.add_constant(X1)).fit() 
print(model3.summary())

model3.condition_number
