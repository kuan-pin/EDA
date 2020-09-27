# Example 5.1
# Condition Number and Correlation Matrix
%cd C:/Course19/ceR/python

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)

model = smf.ols(formula='EM ~ YEAR + PGNP + GNP + AF',data = data).fit() 
print(model.summary())

#condition number 
model.condition_number
# cm=model.eigenvals
# np.sqrt(max(cm)/min(cm))

#correlation matrix 
data[['EM', 'YEAR', 'PGNP', 'GNP', 'AF']].corr()

# alternatively, using design-matrix to compute the condition number
# pip install patsy
import patsy as pt
y, X = pt.dmatrices('EM ~ YEAR + PGNP + GNP + AF',data = data)
np.linalg.cond(X)
