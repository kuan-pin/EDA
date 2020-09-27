# Example 6.3
# Estimating Probability Distributions
# for a given set of data (sample) following a probability distribution
# find the parameters to maximize the probabity (likelihood)

import pandas as pd 
import numpy as np 
import scipy.stats as stats
from scipy import optimize 
from scipy.special import gamma

yed = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/yed20.txt', sep='\s+', nrows=20) 
yed.head() 
y = yed.Income/10 #scale the data may be helpful

#negative normal likelihood: b[0]=mu, b[1]=sigma 
#note that the minus sign is used here, because convex optimization is used to find the minimum
#log-normal likelihood: b[0]=mu, b[1]=sigma 
def llfn(b): 
    return -np.sum(stats.norm.logpdf(y, loc=b[0], scale=b[1])) 

def llfln(b): 
    return -np.sum(np.log(stats.lognorm.pdf(y, s=b[1], scale=np.exp(b[0]))))
"""
def llfln(b): 
    return -np.sum(np.log(stats.norm.pdf(np.log(y), loc=b[0], scale=b[1])*(1/y)))
"""
#negative gamma likelihood: b[0]=rho, b[1]=lambda 
def llfg(b): 
    return -np.sum(np.log(stats.gamma.pdf(y,a=b[0],scale=1/b[1])))

b1 = [3., 2.] 
res1 = optimize.minimize(llfn,b1,method='bfgs')
res1

b2 = [1., .5] 
res2 = optimize.minimize(llfln,b2,method='bfgs')
res2

b3 = [2., 1.] 
res3 = optimize.minimize(llfg,b3,method='bfgs')
res3

# report and print regression output
def print_output(b,vb,bname):
    "Print Regression Output"
    se = np.sqrt(np.diag(vb))
    tr = b/se
    params = pd.DataFrame({'Parameter': b, 'Std. Error': se, 't-Ratio': tr}, index=bname)
    var_cov = pd.DataFrame(vb, index=bname, columns=bname)
    print('\nParameter Estimates')
    print(params)
    print('\nVariance-Covriance Matrix')
    print(var_cov)
    return

from numpy.linalg import inv, det
import statsmodels.tools.numdiff as nd
H = nd.approx_hess(res3.x,llfg)
# check Gradient and Hessian
H  # positive definite?
det(H)
inv(H)  # not necessary the same as res['hess_inv']?
Hinv = inv(H)
var = Hinv    

print_output(res3['x'],var,['x1','x2'])

