# Example 13.5 Klein's Model I Revisited
# Nonliear FIML Estimation, Goldfeld-Quandt (1972), p.34
#
import pandas as pd
import numpy as np
import statsmodels.api as sm
Kdata = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/klein.txt',sep='\s+',nrows=22)
# Year: 1920 -1941 
# C: Consumption in billions of 1934 dollars.
# P: Private profits.
# I: Investment.
# W1: Private wage bill.
# W2: Government wage bill.
# G: Government nonwage spending.
# T: Indirect taxes plus net exports.
# X: Total private income before taxes, or
# X = Y + T - W2 where Y is after taxes income.
# K1: Capital stock in the begining year, or
# capital stock lagged one year. 
# K1[1942]=209.4
P1 = Kdata.P.shift(1)
X1 = Kdata.X.shift(1)
W = Kdata.W1 + Kdata.W2
K = Kdata.K1.shift(-1)
K[21] = 209.4
A = Kdata.YEAR[1:] - 1931

Kdata_ = pd.concat([Kdata, P1, X1, W, K, A], axis=1).dropna()
Kdata_.columns = ['YEAR','C','P','W1','I','K1','X','W2','G','T','P1','X1','W','K','A']
Kdata_ = sm.add_constant(Kdata_)
# redefine G' = G+W2
Kdata_['G'] = Kdata_['G']+Kdata_['W2']

Y = np.array(Kdata_[['P','W1','K']])
X = np.array(Kdata_[['P1','K1','X1','W2','G','T','A']])
Y = Y-Y.mean(axis=0)
X = X-X.mean(axis=0)


from numpy import pi,log,abs
from numpy.linalg import det

def klein1(y,x,c):
    a = c[0:3]
    b = c[3:6]
    r = c[6:9]
    n = x.shape[0]
    beta = np.array([[-1, b[0], r[0]],
            [a[0], -1, 0],
            [a[1], 0, -1]])
    gama = np.array([[a[2], 0, r[1]],
            [-a[1], 0, r[2]],
            [0, b[1], 0],
            [a[0], 0, 0],
            [a[1], 0, 0],
            [-a[1], b[0], 0],
            [0, b[2], 0]])
    u=y@beta+x@gama  # stochastic errors
    # log-likelihood value
    ll=-0.5*n*3*(1+log(2*pi))-0.5*n*log(det(u.T@u/n))+n*log(abs(det(beta)))   
    return(ll)

# using statsmodels' GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModel

class llf_klein1(GenericLikelihoodModel):
    def loglike(self, params):
        b = params
        y = self.endog
        x = self.exog
        return klein1(y,x,b)
    
klein1_model = llf_klein1(Y,X)
b0 = np.array([0.2041,0.1025,0.22967,
               0.72465,0.23273,0.28341,
               0.23116,0.541,0.854])
bname = ['A1','A2','A3','B1','B2','B3','R1','R2','R3']
klein1_results = klein1_model.fit(b0,method='bfgs')
print(klein1_results.summary(xname=bname))
print(klein1_results.cov_params())
