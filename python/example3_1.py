"""
 Example 3.1
 Simple Regression
"""
%cd C:/Course20/ceR/python

import numpy as np
import pandas as pd
# pip install statsmodels
# pip install matplotlib
import statsmodels.api as sm
# data = sm.datasets.longley.load(as_pandas=False)
data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)
PGNP=data['PGNP']
GNP=data['GNP']/1000
EM=data['EM']/1000
RGNP=100*GNP/PGNP

model=sm.OLS(EM,sm.add_constant(RGNP))
results=model.fit()
print(results.summary())

# using R style of formula
df1=pd.concat([EM,RGNP])
model1=sm.OLS.from_formula('EM~RGNP',data=df1)
results1=model1.fit()
print(results1.summary())

import statsmodels.formula.api as smf
model2=smf.ols('EM~RGNP',data=df1)
results2=model2.fit()
print(results2.summary())

import matplotlib.pyplot as plt
%matplotlib inline
EM_fitted = results1.fittedvalues

fig, ax = plt.subplots(figsize=(8, 6)) 
ax.plot(RGNP, EM, 'o', label = 'data') 
ax.plot(RGNP, EM_fitted, 'r-', label = 'OLS') 
ax.legend(loc='best') 
plt.xlabel('RGNP') 
plt.ylabel('EM')
plt.show()

%who   
# clean up
%reset -f
