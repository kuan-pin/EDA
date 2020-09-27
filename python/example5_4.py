# Example 5.4
# Ridge Regression and Principal Components

import numpy as np 
import pandas as pd 
import statsmodels.api as sm

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)
df1 = data[['YEAR', 'PGNP', 'GNP', 'AF']] 

X = sm.add_constant(df1)

model1 = sm.OLS(data.EM, X).fit()
model1.summary()

# ridge regression model using linear algebra
from numpy.linalg import inv
b = np.array(model1.params) 
i = np.eye(5) 
var = model1.cov_params() 
ridge_coef = inv(i + .3*inv(X.T.dot(X))).dot(b) 
ridge_var = (inv(i + .3*inv(X.T.dot(X))).dot(var)).dot(inv(i + .3*inv(X.T.dot(X)))) 
ridge_std = np.sqrt(np.diag(ridge_var))
print('Ridge Regression Model:') 
print(pd.DataFrame({'Coefficient': ridge_coef, 'Std Error': ridge_std}, columns=['Coefficient', 'Std Error'], index=['Constant', 'YEAR', 'PGNP', 'GNP', 'AF']),'\n')
# alternatively, ridge regression may be performed with
# statsmodels.regression.linear_model.OLS.fit_regularized
# or scikit-learn package

# principal components model using linear algebra
from numpy.linalg import eig
r, v = eig(X.T.dot(X)) 
v = v.T[r>0.1] 
v = v.T 
pca_coef = v.dot(v.T).dot(b) 
pca_var = (v.dot(v.T).dot(var)).dot(v.dot(v.T)) 
pca_std = np.sqrt(np.diag(pca_var))
print('Principal Components Model:') 
print(pd.DataFrame({'Coefficient': pca_coef, 'Std Error': pca_std}, columns=['Coefficient', 'Std Error'], index=['Constant', 'YEAR', 'PGNP', 'GNP', 'AF']))
# alternatively, PCA may call
# from statsmodels.multivariate.pca import PCA
