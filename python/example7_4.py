# Example 7.4 
# Box-Cox Variable Transformation 

import numpy as np
import pandas as pd

money = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/money.txt', sep='\s+')
print(money.describe())

m0 = money['Money']/1000  # money
r0 = money['Interest']    # interest rate
y0 = money['GNP']/1000    # income

from numpy import log, sqrt
from scipy import stats, optimize

# log-likelihood function
def llf(b):
    m = (m0 ** b[4] - 1) / b[4]   # money
    r = (r0 ** b[3] - 1) / b[3]   # interest rate
    y = (y0 ** b[3] - 1) / b[3]   # income
    e = m - b[0] - b[1] * r - b[2] * y  # residual
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    lj = log(m0) * (b[4] - 1)  # log jacobian
    return(-sum(ll+lj))

# lambda = 1
def llf1(b):
    m = (m0 ** b[3] - 1) / b[3]   # money
    r = r0   # interest rate
    y = y0   # income
    e = m - b[0] - b[1] * r - b[2] * y
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    lj = log(m0) * (b[3] - 1)  # log jacobian
    return(-sum(ll+lj))

# lambda -> 0
def llf2(b):
    m = (m0 ** b[3] - 1) / b[3]   # money
    r = log(r0)   # log interest rate
    y = log(y0)   # log income
    e = m - b[0] - b[1] * r - b[2] * y
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    lj = log(m0) * (b[3] - 1)  # log jacobian
    return(-sum(ll+lj))

# theta = 1
def llf3(b):
    m = m0    # money
    r = (r0 ** b[3] - 1) / b[3]   # interest rate
    y = (y0 ** b[3] - 1) / b[3]   # income
    e = m - b[0] - b[1] * r - b[2] * y  # residual
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    return -sum(ll)

# theta -> 0
def llf4(b):
    m = log(m0)   # log money
    r = (r0 ** b[3] - 1) / b[3]   # interest rate
    y = (y0 ** b[3] - 1) / b[3]   # income
    e = m - b[0] - b[1] * r - b[2] * y  # residual
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    lj = log(1/m0)
    return(-sum(ll+lj))

# lambda = theta
def llf5(b):
    m = (m0 ** b[3] - 1) / b[3]   # money
    r = (r0 ** b[3] - 1) / b[3]   # interest rate
    y = (y0 ** b[3] - 1) / b[3]   # income
    e = m - b[0] - b[1] * r - b[2] * y  # residual
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    lj = log(m0) * (b[3] - 1) # log jacobian
    return(-sum(ll+lj))

# lambda = theta = 1
def llf6(b):
    m = m0   # money
    r = r0   # interest rate
    y = y0   # income
    e = m - b[0] - b[1] * r - b[2] * y  # residual
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    return -sum(ll)

# lambda = theta = 0
def llf7(b):
    m = log(m0)   # log money
    r = log(r0)   # log interest rate
    y = log(y0)   # log income
    e = m - b[0] - b[1] * r - b[2] * y
    ll = log(stats.norm.pdf(e, loc=0, scale=sqrt((e**2).mean()))) 
    lj = log(1/m0)
    return(-sum(ll+lj))
    
b1 = [1, .5, -1, -1, 1]
M = optimize.minimize(llf,x0=b1,method='BFGS')
ll = -M['fun']

b2 = M.x[:-1] # [1, .5, -1, -1]
M1 = optimize.minimize(llf1,x0=b2,method='BFGS')
ll1 = -M1['fun']

M2 = optimize.minimize(llf2,x0=b2,method='BFGS')
ll2 = -M2['fun']

M3 = optimize.minimize(llf3,x0=b2,method='BFGS')
ll3 = -M3['fun']

M4 = optimize.minimize(llf4,x0=b2,method='BFGS')
ll4 = -M4['fun']

M5 = optimize.minimize(llf5,x0=b2,method='BFGS')
ll5 = -M5['fun']

b3 = M.x[:3]  # [1, .5, -1]
M6 = optimize.minimize(llf6,x0=b3,method='BFGS')
ll6 = -M6['fun']

M7 = optimize.minimize(llf7,x0=b3,method='BFGS')
ll7 = -M7['fun']

lli = [ll, ll1, ll2, ll3, ll4, ll5, ll6, ll7]
params = ['lambda!=theta', 'lambda=1', 'lambda->0', 'theta=1','theta->0', 
          'lambda=theta', 'lambda=theta=1', 'lambda=theta->0']

lrtest = list()
for i in lli:
    lrtest.append(-2*(i-ll))
    
table = pd.DataFrame({'Parameter':params, 'Log Likelihood':lli, 
                     'Likelihood Ratio Test':lrtest},
                     index = ['M','M1','M2','M3','M4','M5','M6','M7'])
print(table)
