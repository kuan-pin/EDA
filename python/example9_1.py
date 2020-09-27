# Example 9.1 Heteroscedasticity-Consistent Variance-Covariance Matrix

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/greene.txt',sep='\s+',nrows=51)
data = data.dropna()
print(data.describe())

spending = data['SPENDING']
income = data['INCOME']/10000
df1 = pd.concat([spending,income],keys=['spending','income'],axis=1)

model = smf.ols('spending~income+I(income**2)',data = df1)
model1 = model.fit()  # OLS
print(model1.summary())
print('Variance-Covariance Matrix of Coefficients')
print(model1.cov_params())
print('Correlation Matrix of Coefficents')
print(model1.cov_params() / model1.bse / model1.bse[:, np.newaxis])

# assuming heteroscedasticity (White 1980)
model2 = model.fit(cov_type='HC0')
# model2 = model1.get_robustcov_results(cov_type='HC0')
print(model2.summary())
print('Variance-Covariance Matrix of Coefficients')
print(model2.cov_params())  # same as (model1.cov_HC0)
print('Correlation Matrix of Coefficents')
print(model2.cov_params() / model2.bse / model2.bse[:, np.newaxis])

# compare different types of VCV
bsedf = pd.DataFrame({'bse':model1.bse,
                      'HC0':model1.HC0_se,
                      'HC1':model1.HC1_se,
                      'HC2':model1.HC2_se,
                      'HC3':model1.HC3_se})
print(bsedf)
