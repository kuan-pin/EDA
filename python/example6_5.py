# Example 6.5 Minimizing Sum-of-Squares Function
# Estimating a CES Production Function

import pandas as pd 
import numpy as np
from scipy import optimize

judge = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/judge.txt', sep='\s+', names=['l','k','q'])
print(judge.describe())
l, k, q = judge.l, judge.k, judge.q

def ces(b):
    e = np.log(q)-(b[0]+b[3]*np.log(b[1]*l**b[2]+(1-b[1])*k**b[2]))
    return e

# objective function to be minimized
def sse(b):
    e = ces(b)
    return np.sum(e**2)

b0 = [1., .5, -1., -1.]
res = optimize.minimize(sse, b0, method='Nelder-Mead')
res = optimize.minimize(sse, res.x, method='BFGS')
res

from numpy.linalg import inv
import statsmodels.tools.numdiff as nd
# using numeric derivatives to compute jacobian
gv = nd.approx_fprime(res.x,ces)
G = (gv.T @ gv) 
H = nd.approx_hess(res.x,sse)
# inv(H) is more reliable than res.hess_inv

print('Final Result:')
print('Function Value =', res['fun'])
print('Parameters =', res['x'])
print('Gradient Vector =', res['jac'])
print('Hessian Matrix = \n', inv(res['hess_inv']))

# assuming homoscedastic normally distributed errors
# via Information Matrix Equality
# hessian matrix may be approximated by (gv.T @ gv)

v = res['fun']/len(judge) # variance of residuals, sse/n
var = v*inv(gv.T @ gv)
var = v*res['hess_inv']
var = v*inv(H)
se = np.sqrt(np.diag(var))
tr = res['x']/se
# print parameters and statistics
params = pd.DataFrame({'Parameter': res.x, 'Std. Error': se, 't-Ratio': tr}, 
                   index=['b1','b2','b3','b4'])
print(params)

# print variance-covariance matrix
var_cov = pd.DataFrame(var, index=['b1','b2','b3','b4'], columns=['b1','b2','b3','b4'])
print('Asymptotic Variance-Covariance Matrix')
print(var_cov)

# alternatively, using optimize.leastsq or optimize.least_squares
# using lmfit package may be easier, see Example 7_1
