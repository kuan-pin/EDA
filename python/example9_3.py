# Example 9.3 Breusch-Pagan and White Tests

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
%matplotlib inline

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/greene.txt',sep='\s+',nrows=51)
data = data.dropna()  # take care of missing obs
print(data.describe())

spending = data['SPENDING']
income = data['INCOME']/10000

df1 = pd.concat([spending,income],keys=['spending','income'],axis=1)
model1 = smf.ols(formula='spending~income+I(income**2)',data = df1).fit()
print(model1.summary())

resid1 = model1.resid
# QQ-Plot of residuals
fig1 = sm.qqplot(resid1,fit=True,line='45')
plt.show()

# Jarque-Bera Wald Test for Normality
jbtest = sms.stattools.jarque_bera(resid1)
print('Jarque-Bera Wald Test for Normality')
print('Skewness of Residuals = ', jbtest[2])
print('Kurtosis of Residuals = ', jbtest[3])
df2 = pd.Series({'Chi-Sq( 2)':jbtest[0], 'Prob>Chi-Sq':jbtest[1]})
print(df2)

# Breusch-Pagan Test for Heteroscedasticity
bptest = sms.diagnostic.het_breuschpagan(resid1,model1.model.exog)
# White Test for Heteroscedasticity, including squares and cross-product of exog
white_test = sms.diagnostic.het_white(resid1,model1.model.exog)

print('Breusch-Pagan Test and White LM Tests for Heteroscedasticity')
df3 = pd.DataFrame({'Test Type':['Koenkar-Basset Test', 'White Test'],
                   'Chi-Sq':[bptest[0], white_test[0]], 'DF':[2, 4],
                   'Prob>Chi-Sq':[bptest[1], white_test[1]]})
print(df3)
