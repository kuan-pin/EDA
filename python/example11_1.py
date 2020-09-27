# Example 11.1 Lagged Dependent Variable Model
# Estimation and Testing for Autocorrelation 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm 

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",index_col='YEAR',sep='\s+',nrows=66)
y = data['Y']
c = data['C']
c1 = c.shift(1) # the first lag of the dependent variable
df = pd.DataFrame({"y":y,"c":c,"c1":c1})

ols1 = sm.OLS.from_formula('c~c1+y',df).fit()
print(ols1.summary())
print(anova_lm(ols1))

def reg_anova(model):
    res = pd.DataFrame({'SS': [model.ess, model.ssr, model.centered_tss],
                        'DF': [model.df_model, model.df_resid, model.nobs - 1],
                        'MSS': [model.mse_model, model.mse_resid, model.mse_total],
                        'F': [model.fvalue, np.nan, np.nan],
                        'Prob>F': [model.f_pvalue, np.nan, np.nan]},
                       index=['Explained', 'Residual', 'Total'])
    return res

print(reg_anova(ols1))

#from statsmodels.stats.anova import AnovaResults
#print(AnovaResults(reg_anova(ols1)))# anova results

DW = sm.stats.durbin_watson(ols1.resid)
print('Durbin-Watson Test Statistic =',DW)

# Durbin-H test
n = ols1.nobs
var_c1 = ols1.cov_params().at['c1','c1']
# var_c1 = ols1.bse['c1']**2
DH = (1-DW/2)*np.sqrt(n/(1-n*var_c1))
print("Durbin-H Test Statistic",DH) 
# p-value of DH
# import scipy
# scipy.stats.norm.cdf(DH)

# cochrane-orcutt / prais-winsten with given AR(1) rho, 
# derived from ols model, default to cochrane-orcutt 
def ols_ar1(model,rho,drop1=True):
    x = model.model.exog
    y = model.model.endog
    ystar = y[1:]-rho*y[:-1]
    xstar = x[1:,]-rho*x[:-1,]
    if drop1 == False:
        ystar = np.append(np.sqrt(1-rho**2)*y[0],ystar)
        xstar = np.append([np.sqrt(1-rho**2)*x[0,]],xstar,axis=0)
    model_ar1 = sm.OLS(ystar,xstar).fit()
    return(model_ar1)

# cochrane-orcutt / prais-winsten iterative procedure
# default to cochrane-orcutt (drop1=True)
def OLSAR1(model,drop1=True):
    x = model.model.exog
    y = model.model.endog
    e = y-x@model.params
    e1 = e[:-1]; e0 = e[1:]
    rho0 = np.dot(e1,e[1:])/np.dot(e1,e1)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1 = ols_ar1(model,rho0,drop1)
        e = y - (x @ model1.params)
        e1 = e[:-1]; e0 = e[1:]
        rho1 = np.dot(e1,e[1:])/np.dot(e1,e1)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        print('Rho = ', rho0)
    # pint final iteration
    # print(sm.OLS(e0,e1).fit().summary())
    model1 = ols_ar1(model,rho0,drop1)
    return(model1)

# AR(1) based on cochrane-orcutt iterative procedure   
ar1_co = OLSAR1(ols1)
# ar1_co = OLSAR1(model_ols,drop1=True)
print(ar1_co.summary())

# AR(1) based on prais-winsten iterative procedure
ar1_pw = OLSAR1(ols1,drop1=False)
print(ar1_pw.summary())
# the results are based on transformed model

# Breusch-Godfrey test
print('Breusch-Godfrey test')
for i in range(4):
    BG_lm, BG_lmpval, _, _ = sm.stats.diagnostic.acorr_breusch_godfrey(ar1_pw, nlags=i+1)
    print('LM test = %f, df = %d, p-value = %f' %(BG_lm, i+1, BG_lmpval))

e = ar1_pw.resid
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(e,zero=False)
plot_pacf(e,zero=False)    

