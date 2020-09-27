# Example 10.1 Heteroscedasticity Autocorrelation
# Consistent Variance-Covariance Matrix

import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', nrows=39)

X = np.log(data.X)
L = np.log(data.L1)
K = np.log(data.K1)
df = pd.DataFrame({'X': X, 'L': L, 'K': K})

model = sm.OLS.from_formula('X~L+K',data=df)
# OLS
model1 = model.fit()
print(model1.summary())
print('Variance-Covariance Matrix of Coefficients')
print(model1.cov_params())
print('Correlation Matrix of Coefficients')
print(model1.cov_params() / model1.bse / model1.bse[:, np.newaxis])

# HAC(4)
model2 = model.fit(cov_type='HAC',cov_kwds={'maxlags':4})
print(model2.summary())# ols results
print('Variance-Covariance Matrix of Coefficients')
print(model2.cov_params())
print('Correlation Matrix of Coefficients')
print(model2.cov_params() / model2.bse / model2.bse[:, np.newaxis])

# compare estimated s.e. under OLS and HAC(4)
df1 = pd.DataFrame({'b':model1.params,'bse1':model1.bse,'bse2':model2.bse})
print(df1)

# HAC(4) same as model2
model3 = model1.get_robustcov_results(cov_type='HAC', maxlags=4)
print(model3.summary())# ols results
print('Variance-Covariance Matrix of Coefficients')
print(model3.cov_params())
print('Correlation Matrix of Coefficients')
print(model3.cov_params() / model3.bse / model3.bse[:, np.newaxis])

# for comparison, using sanwich_covariance
# sms.sandwich_covariance.cov_white_simple(model1)  # HC0
sm.stats.sandwich_covariance.cov_hac_simple(model1,nlags=4)  # HAC(4)
sm.stats.sandwich_covariance.cov_hac(model1,nlags=4)
sm.stats.sandwich_covariance.cov_hac(model1,nlags=4,use_correction=False)  # HAC(4)
# same as model2 and model3
