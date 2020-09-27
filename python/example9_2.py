# Example 9.2 Goldfeld-Quandt Test and Correction for Heteroscedasticity

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/greene.txt',sep='\s+',nrows=51)
data = data.dropna()
print(data.describe())

spending = data['SPENDING']
income = data['INCOME']/10000

df1 = pd.concat([spending,income],keys=['spending','income'],axis=1)

df1 = df1.sort_values(by='income') # sort income in ascending order
model1 = smf.ols(formula='spending~income+I(income**2)',data = df1[0:17]).fit()
model2 = smf.ols(formula='spending~income+I(income**2)',data = df1[33:50]).fit()

# Goldfeld-Quandt Test
rss1 = model1.ssr
df_resid1 = model1.df_resid

rss2 = model2.ssr
df_resid2 = model2.df_resid

GQstat=(rss2/df_resid2)/(rss1/df_resid1)
print("Goldfeld-Quandt Test Statistic = ",GQstat)
# p-value (Prob>F)
1-stats.f.cdf(GQstat,df_resid2,df_resid1)

# Weighted Least Squares
model3 = smf.wls(formula='spending~income+I(income**2)',data = df1,
                 weights=1./df1['income']**2).fit()
print(model3.summary())

# alternatively, using statsmodels.stats.diagnostic.het_goldfeldquandt
# data is splitted as (0:17,33:50), dropping 16 obs in the middle
import statsmodels.api as sm
import statsmodels.stats.api as sms
y = data['SPENDING']
x1 = data['INCOME']/10000
x2 = x1**2
X = pd.concat([x1,x2],axis=1)
X = sm.add_constant(X)
# sort data according to income in ascending order
# drop data in the middle: [0:split and split+drop:}
GQ_test = sms.diagnostic.het_goldfeldquandt(y,X,idx=1,split=17,drop=16)
df2 = pd.Series({'F(14,14)':GQ_test[0], 'Prob>F':GQ_test[1]})
print(df2)
