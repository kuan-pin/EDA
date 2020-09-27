# Example 6.6 Maximizing Log-Likelihood Function
# Estimating a CES Production Function

import pandas as pd 
import numpy as np
from scipy import stats, optimize

judge = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/judge.txt',sep='\s+', names=['l','k','q'])
print(judge.describe())
l, k, q = judge.l, judge.k, judge.q

# objective function to be maximized (log-normal likelihood function)
def llfobs(b):
    e = np.log(q) - b[0] - b[3]*np.log(b[1]*l**b[2]+(1-b[1])*k**b[2])
    ll = np.log(stats.norm.pdf(e, loc=0, scale=b[4]))
    return ll+np.log(1/q)  # include log-jacobian

def llf(b):
    llf = llfobs(b)
    return -np.sum(llf)   # negative value for minimization

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

# maximum Likelihood
b0 = [1., .5, -1., -1., 1.]
res = optimize.minimize(llf, b0, method='Nelder-Mead')
res = optimize.minimize(llf, res['x'], method='bfgs')

from numpy.linalg import inv
import statsmodels.tools.numdiff as nd
# using numeric derivatives to compute jacobian
gv = nd.approx_fprime(res.x,llfobs)
G = (gv.T @ gv) 
H = nd.approx_hess(res.x,llf)
# inv(H) is more reliable than res.hess_inv

print('Final Result:')
print('Function Value =', -res['fun'])
print('Parameters =', res['x'])
print('Gradient Vector =', -res['jac'])
print('Hessian Matrix = \n', -inv(res['hess_inv']))

Hinv = inv(H)  # more reliable than res['hess_inv']
var1 = Hinv    # inverse of hessian
var2 = inv(G)  # inv(G/len(judge))/len(judge), OPG var-cov
var3 = (Hinv @ G @ Hinv)  # robust var-cov
var1 # var1=var2 or -H=G if Information Matrix Equality holds
var2 # standard var-cov matrix
var3 # robust var-cov matrix

var = var1 # try var2, var3, etc.   
print_output(res['x'],var,['b1','b2','b3','b4','b5'])

# alternatively, using statsmodels' GenericLikelihoodModel
# see Example7_2
