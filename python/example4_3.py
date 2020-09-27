# Example 4.3
# Testing for Structural Change

import numpy as np 
import pandas as pd 
from scipy import stats 
import statsmodels.formula.api as smf 

data_cjx=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt',sep='\s+',nrows=39)
break_year = (data_cjx['YEAR'] > 1948).astype(int)

X = np.log(data_cjx['X']) 
L = np.log(data_cjx['L1'])
K = np.log(data_cjx['K1'])
df = pd.concat([X, L, K], keys=['X', 'L', 'K'], axis=1) 

# restricted model
model1 = smf.ols(formula='X ~ L + K', data=df).fit()
print(model1.summary())

# unrestricted model
model2 = smf.ols(formula='X ~ (L + K)*break_year', data=df).fit()
print(model2.summary())

DFr = model1.df_resid
RSSr = model1.ssr
DFur = model2.df_resid
RSSur = model2.ssr

# F test for structural change
Ftest = ((RSSr-RSSur)/(DFr-DFur))/(RSSur/DFur)
Ftest
# Calculate P-value
1-stats.f.cdf(Ftest, DFr-DFur, DFur)
# threshold = stats.f.ppf(0.95, DFr-DFur, DFur)

# alternatively, do ANOVA
from statsmodels.stats.anova import anova_lm 
anova_lm(model1,model2)

