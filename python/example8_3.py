# Example 8.3 Tobit Analysis of Extramarital Affairs
# Tobit (Tobin's Logit) Regression Model
# Analysis of Extramarital Affairs (Fair, 1978)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

fair = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/fair.txt', sep='\s+')
fair = fair.dropna()
print(fair.describe())

X = fair[['Z2','Z3','Z5','Z7','Z8']]
y = fair['Y']
X = sm.add_constant(X,prepend=False)  # constant added to the last

# Log-likelihood function: tobit model
def llfobs(b):
    e = y - (X @ b[:-1])
    ll0 = stats.norm.logcdf(e[y==0],0,b[-1])
    ll1 = stats.norm.logpdf(e[y>0],0,b[-1])
    return np.concatenate([ll0,ll1])
 
def llf(b):
    ll = llfobs(b)
    return (-np.sum(ll))

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
    
b = [0., 0., 0., 0., 0., 0., 1.]
# convergence is not easy
res = optimize.minimize(llf, b, method='Nelder-Mead')
res = optimize.minimize(llf, res.x, method='BFGS')
res

print('Final Result:')
print('Log Likelihood =', -res['fun'])
print('Parameters =', res['x'])
print('Gradient Vector =', -res['jac'])

from numpy.linalg import inv
print('Hessian Matrix = \n', -inv(res['hess_inv']))

# print parameters
import statsmodels.tools.numdiff as nd
Hinv = inv(nd.approx_hess(res['x'],llf))  # more reliable than res['hess_inv']
var = Hinv

print_output(res['x'],var,['Z2','Z3','Z5','Z7','Z8','const','sigma'])

# using statsmodels' GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModel

class llf_tobit(GenericLikelihoodModel):
    def loglikeobs(self, params):
        b = params
        return llfobs(b)

tobit_model = llf_tobit(y,X)  # data input is a dummy
tobit_results = tobit_model.fit(b,method='bfgs')
print(tobit_results.summary())

