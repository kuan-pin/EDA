# Example 7.3 Hypothesis Testing for Nonlinear Models
# Hypothesis Testing 

import numpy as np
import pandas as pd
from scipy import stats, optimize
from numpy import log, sqrt, mean
from numpy.linalg import inv
import statsmodels.tools.numdiff as nd

judge = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/judge.txt',sep='\s+', names=['l','k','q'])
print(judge.describe())
l, k, q = judge.l, judge.k, judge.q

# initial parameter values
# b0=[1,0.5,-1,-1]
b0=[1,0.5,-1]

# log-likelihood function of unretricted CES model (4 parameters)
def llfobs(b):
    e = log(q)-b[0]-b[3]*log(b[1]*l**b[2]+(1-b[1])*k**b[2])
    s = sqrt(mean(e**2))
    ll = log(stats.norm.pdf(e, loc=0, scale=s)) # log pdf
    return ll

def llf(b):
    ll = llfobs(b)
    return (-np.sum(ll))

# log-likelihood function of retricted CES model (3 parameters)
# restrictions: b[3]=1/b[2]
def llfobsr(b):
    e = log(q)-b[0]-(1/b[2])*log(b[1]*l**b[2]+(1-b[1])*k**b[2])
    s = sqrt(mean(e**2))
    ll = log(stats.norm.pdf(e, loc=0, scale=s)) # log pdf
    return (ll)

def llfr(b):
    ll = llfobsr(b)
    return (-np.sum(ll))

# constraint function: b[3]=1/b[2]
def cf(b):  # constraint function
    return b[3]*b[2]-1


# Lagrangian Multiplier Test: based on constrained model
M0 = optimize.minimize(llfr,x0=b0,method='BFGS')
M0
ll0 = -M0['fun']
b1 = np.append(M0['x'],1/M0['x'][2]) # original parameterization
dll = nd.approx_fprime(b1,llfobs)
# using numeric derivatives to compute 
# jacobian (gradient matrix for vector-valued llf
gll = np.sum(dll,axis=0)  # gradient vector of llf
# use OPG to approximate var-cov(dll)
lmtest = gll.T @ inv(dll.T @ dll) @ gll
lmtest
1-stats.chi2.cdf(lmtest,1)

# Wald Test: based on unconstrained model
M1 = optimize.minimize(llf,x0=b1,method='Nelder-Mead')
M1 = optimize.minimize(llf,x0=M1.x,method='BFGS')
M1
ll1 = -M1['fun']
beta1 = M1['x']
vcov1 = inv(nd.approx_hess(beta1,llf)) # vcov1 = M1['hess_inv']
# hess_inv may not reliable here, or use OPG to approximate
# dll1 = nd.approx_fprime(beta1,llf)
# vcov1 = inv(dll1.T @ dll1)
c1 = np.array([cf(beta1)])
gc1 = nd.approx_fprime(beta1,cf).reshape(1,-1)
wtest = c1.T @ inv(gc1 @ vcov1 @ gc1.T) @ c1
wtest
1-stats.chi2.cdf(wtest,1)

lrtest = -2*(ll0-ll1)
lrtest
1-stats.chi2.cdf(lrtest,1)

test = {'Wald Test':wtest,
       'Lagrangian Multiplier Test':lmtest,
       'Likelihood Ratio Test':lrtest}
test = pd.Series(test)
print(test)
