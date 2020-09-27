# Example 6.4
# Mixture of Two Normal Distributions

import pandas as pd 
import numpy as np
import scipy.stats as stats
from scipy import optimize

yed = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/yed20.txt', sep='\s+', nrows=20) 
y = yed.Income/10
#mixture of two normal distributions 
#b[0]=mu1, b[1]=sigma1, b[2]=mu2, b[3]=sigma2, b[4]=lambda 
def llf(b): 
    lf1 = stats.norm.pdf(y, loc=b[0], scale=b[2]) 
    lf2 = stats.norm.pdf(y, loc=b[1], scale=b[3]) 
    return -np.sum(np.log((1-b[4])*lf1 + b[4]*lf2))

b = [3, 3, 2, 2, .5] 
res1 = optimize.minimize(llf,b,method='Nelder-Mead')
b = res1.x
res2 = optimize.minimize(llf,b,method='bfgs')
print(res2)

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
H = nd.approx_hess(res2.x,llf)
# check Gradient and Hessian
H  # positive definite?
det(H)
inv(H)  # not necessary the same as res['hess_inv']?
Hinv = inv(H)
var = Hinv    
# var = res2.hess_inv
print_output(res2['x'],var,['x1','x2','x3','x4','x5'])

