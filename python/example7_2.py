# Example 7.2 Box-Cox Transformation
# U. S. Money Demand Equation
# Maximum Likelihood Estimation
# Greene: Chapter 7

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

money = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/money.txt', sep='\s+')
print(money.describe())

m0 = money['Money']/1000  # money
r0 = money['Interest']    # interest rate
y0 = money['GNP']/1000    # income
data = pd.concat([m0,r0,y0],keys=['m','r','y'],axis = 1)

model0 = smf.ols(formula = 'm~r+y',data = data)
res0 = model0.fit()
print(res0.summary())

b0 = res0.params  # initial parameters
b0 = list(b0)+[1,1]

# log-likelihood function
from scipy import stats, optimize

def rf(b):
    # box-cox transformation
    m = (m0 ** b[4] - 1) / b[4]   # money
    r = (r0 ** b[3] - 1) / b[3]   # interest rate
    y = (y0 ** b[3] - 1) / b[3]   # income
    e = m - b[0] - b[1] * r - b[2] * y  # residual
    return e

def llfobs(b):
    e = rf(b)
    s = np.sqrt((e**2).mean())
    ll = stats.norm.logpdf(e, loc=0, scale=s) # log pdf
    lj = np.log(m0) * (b[4]-1) # log jacobian
    return(ll+lj)
    
def llf(b):
    return(-sum(llfobs(b)))  # negative sum of log-likelihoods

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

# res = optimize.minimize(llf, b0, method='powell')
# res = optimize.minimize(llf, b0, method='cg', options={'maxiter':5000})
res = optimize.minimize(llf, b0, method='bfgs')
res = optimize.minimize(llf, res.x, method='bfgs')
res # convergence is difficult

print('Final Result:')
# e = rf(res['x'])/stats.gmean(np.exp(jf(res['x'])))  # correction done
# sse = e.dot(e)
# print('Sum of Squares =', sse) # the result on the book is 0.11766
print('Log Likelihood =', -res['fun'])
print('Gradient of Log Likelihood =', -res['jac'])
from numpy.linalg import inv
print('Hessian Matrix = \n', -inv(res['hess_inv']))

# computationof var-cov matrix and se of the parameters
# depends onthe precision of the inverse of hessian matrix
# needto be sure that the optimization achieves convergence successfully

from numpy.linalg import inv, det
import statsmodels.tools.numdiff as nd
# using numeric derivatives to compute jacobian
gv = nd.approx_fprime(res.x,llfobs)
G = (gv.T @ gv) 
H = nd.approx_hess(res.x,llf)
# check Gradient and Hessian
H  # positive definite?
det(H)
inv(H)  # not the same as res['hess_inv']
Hinv = inv(H)
# robust var-cov
var = Hinv @ G @ Hinv 
# or, simply
# var = Hinv    
# var = inv(G)
# var1=var2 or -H=G if Information Matrix Equality holds

# print parameters and var-cov matrix
# s.e. ia not thesame as in thebook?
print_output(res['x'],var,['b1','b2','b3','b4','b5'])

# alternatively, using statsmodels' GenericLikelihoodModel
from statsmodels.base.model import GenericLikelihoodModel

class llf_boxcox(GenericLikelihoodModel):
    def loglikeobs(self, params):
        b = params
        return llfobs(b)

boxcox_model = llf_boxcox(data)
boxcox_results = boxcox_model.fit(b0,method='bfgs')
print(boxcox_results.summary())
# s.e. is not the same as in the book?
