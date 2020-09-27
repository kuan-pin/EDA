"""
# Example 3.5
# Testing for Structural Change
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt',sep='\s+',nrows=39)

year=data['YEAR']
X=np.log(data['X'])
L=np.log(data['L1']) 
K=np.log(data['K1'])

df = pd.concat([X, L, K], keys=['X', 'L', 'K'], axis=1) 
model = smf.ols(formula='X ~ L + K', data=df).fit()
print(model.summary())
model1 = smf.ols(formula='X ~ L + K', data=df, subset=year<1949).fit()
print(model1.summary())
model2 = smf.ols(formula='X ~ L + K', data=df, subset=year>1948).fit()
print(model2.summary())

# from statsmodels.iolib.summary2 import summary_col
# summary_col([model,model1,model2],stars=True,info_dict={"N":lambda x:(x.nobs)})

DFr = model.df_resid
RSSr = model.ssr
DF1 = model1.df_resid
RSS1 = model1.ssr
DF2 = model2.df_resid
RSS2 = model2.ssr
DFur = DF1+DF2
RSSur = RSS1+RSS2

# Chow test
Ftest = ((RSSr-RSSur)/(DFr-DFur))/(RSSur/DFur)
Ftest
# Calculate P-value
1-stats.f.cdf(Ftest, DFr-DFur, DFur)
# threshold = stats.f.ppf(0.95, DFr-DFur, DFur)

