# Example 10.2 Tests for Autocorrelation

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# read data with index_col
data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', index_col='YEAR', nrows=39)

X = np.log(data.X)
L = np.log(data.L1)
K = np.log(data.K1)
df = pd.DataFrame({'X': X, 'L': L, 'K': K})

# OLS
model1 = sm.OLS.from_formula('X~L+K', df).fit()
print(model1.summary())
# plot residuals and check for normality
resid1 = model1.resid
resid1.plot(); plt.show()
resid1.hist(); plt.show()

# Durbin-Watson test for AR(1)
sm.stats.durbin_watson(resid1)

# Jung-Box and Box-Pierce tests for AR(lags)
lags =12  # required for the following tests
jb, jbpv, bp, bppv = sm.stats.diagnostic.acorr_ljungbox(resid1,lags,True)
# print(sm.stats.diagnostic.acorr_ljungbox(resid1,lags,True,return_df=True))
JB_test = pd.DataFrame({'Jung-Box': jb, 
                    'JB-P-val': jbpv,
                   'Box-Pierce': bp, 
                   'BP-P-val': bppv},
                   index=range(1,lags+1))
print(JB_test)

# Breusch-Godfrey test
bgx=np.zeros(lags)
bgxpv=np.zeros(lags)
bgf=np.zeros(lags)
bgfpv=np.zeros(lags)
for i in range(lags):
    bgx[i],bgxpv[i],bgf[i],bgfpv[i] = sm.stats.diagnostic.acorr_breusch_godfrey(model1,i+1)
BG_test = pd.DataFrame({'Chi-Sq': bgx, 'Prob>Chi-Sq': bgxpv,
                        'F': bgf, 'Prob>F': bgfpv}, index=range(1,lags+1))
print(BG_test)
    
# check for residual autocorrelation ACF and PACF
import statsmodels.graphics.tsaplots as tsaplt
import statsmodels.tsa.api as tsa

acf, acf_confint, lbq, lbqpv = tsa.acf(resid1, nlags=lags, alpha=0.05, qstat=True, fft=False)
pacf, pacf_confint = tsa.pacf(resid1, nlags=lags, alpha=0.05)
acf_se = (acf_confint[:,1]-acf)/1.96
pacf_se = (pacf_confint[:,1]-pacf)/1.96

ACF_table = pd.DataFrame({'AR': acf[1:], 'AR_se': acf_se[1:], 
                          'PAR': pacf[1:], 'PAR_se': pacf_se[1:], 
                          'Ljung-Box Q': lbq, 'Q_pval': lbqpv},
                    index=range(1,lags+1))
print(ACF_table)

# plots of ACF and Pacf
tsaplt.plot_acf(model1.resid, lags=lags, zero=False)
tsaplt.plot_pacf(model1.resid, lags=lags, zero=False)
