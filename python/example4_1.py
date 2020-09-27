# Example 4.1
# Seasonal Dummy Variables

import numpy as np 
import pandas as pd 
from scipy import stats 
import statsmodels.formula.api as smf 

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/almon.txt',sep='\s+',nrows=60)

cexp = data['CEXP'] 
capp = data['CAPP'] 
qt = pd.get_dummies(data['YEARQT'] % 10, prefix='Q') 
df = pd.concat([cexp,capp,qt],axis = 1)
df.head()

model = smf.ols('CEXP ~ CAPP + Q_1 + Q_2 + Q_3',data = df).fit() 
print(model.summary())
# hypothsis testing
h1 = 'Q_1=0, Q_2=0, Q_3=0'
model.wald_test(h1,use_f=False)
model.f_test(h1)

# constrained or restricted model
modelr = smf.ols('CEXP ~ CAPP',data = df).fit() 
print(modelr.summary())

DFr = modelr.df_resid
RSSr = modelr.ssr
DFur = model.df_resid
RSSur = model.ssr
# F test
Ftest = ((RSSr-RSSur)/(DFr-DFur))/(RSSur/DFur)
Ftest
# Calculate P-value
1-stats.f.cdf(Ftest, DFr-DFur, DFur)
# threshold = stats.f.ppf(0.95, DFr-DFur, DFur)

from statsmodels.stats.anova import anova_lm 
anova_lm(modelr,model) 

              
           