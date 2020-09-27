# Example 10.6 ARMA(1,1) Error Structure

import numpy as np
import pandas as pd
import statsmodels.api as sm 
from statsmodels.tsa.arima_model import ARMA

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/cjx.txt', sep='\s+', nrows=39)

X = np.log(data.X)
L = np.log(data.L1)
K = np.log(data.K1)
Z = pd.concat([L,K],axis=1)
# Z = sm.add_constant(Z)

ar1 = ARMA(X,exog=Z,order=(1,0)).fit()
print(ar1.summary())

ma1 = ARMA(X,exog=Z,order=(0,1)).fit()
print(ma1.summary())

arma1 = ARMA(X,exog=Z,order=(1,1)).fit()
print(arma1.summary())

# try different estimation method, such as method='mle'
# try higher order of ARMA(p,q)

# check for residual autocorrelation of the final model
e = arma1.resid
lags = 5
bptest = sm.stats.diagnostic.acorr_ljungbox(e, lags=lags, boxpierce=True, return_df=True)
print(bptest)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(e,zero=False)
plot_pacf(e,zero=False)
