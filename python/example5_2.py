# Example 5.2
# Theil's Measure of Multicollinearity
%cd C:/Course19/ceR/python

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)

model = smf.ols(formula='EM ~ YEAR + PGNP + GNP + AF',data = data).fit() 
print(model.summary())
R2 = model.rsquared 
R21 = smf.ols(formula='EM ~ PGNP + GNP + AF',data = data).fit().rsquared 
R22 = smf.ols(formula='EM ~ YEAR + GNP + AF',data = data).fit().rsquared 
R23 = smf.ols(formula='EM ~ YEAR + PGNP + AF',data = data).fit().rsquared 
R24 = smf.ols(formula='EM ~ YEAR + PGNP + GNP',data = data).fit().rsquared
theil = R2 - (R2 - R21) - (R2 - R22) - (R2 - R23) - (R2 - R24) 
print("Thiel's Measure of Multicollinearity = %f" % theil) 
