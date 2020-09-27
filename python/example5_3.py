# Example 5.3
# Lesson 5.3: Variance Inflation Factors (VIF)
import numpy as np
import pandas as pd 
import statsmodels.api as sm

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)

def vif(df, col): 
    k_vars = df.shape[1] 
    x_i = df[col] 
    x_noti = df.drop([col], axis=1) 
    X = sm.add_constant(x_noti) 
    r_squared_i = sm.OLS(x_i, X).fit().rsquared 
    vif = 1 / (1 - r_squared_i)
    vif_list = [col, r_squared_i, vif] 
    vif_df = pd.DataFrame(vif_list, index=['Variable', 'R-Squared', 'VIF']).T 
    return vif_df

df1 = data[['YEAR', 'PGNP', 'GNP', 'AF']] 
vif_table = pd.concat([vif(df1, col) for col in df1.columns], ignore_index=True) 
print(vif_table)

X = sm.add_constant(df1) 
model1 = sm.OLS(data.EM, X).fit()
model1.summary()

# alternatively using statsmodels.states.outliners_influence
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xm = np.array(X)  # index 0 is constant
VIF1 = variance_inflation_factor(Xm,1)
VIF2 = variance_inflation_factor(Xm,2)
VIF3 = variance_inflation_factor(Xm,3)
VIF4 = variance_inflation_factor(Xm,4)
VIF = np.array([VIF1,VIF2,VIF3,VIF4])
VIF
