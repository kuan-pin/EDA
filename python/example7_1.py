"""
# Example 7.1 CES Production Function Revisited
# Estimating CES Production Function
# Judge, et. al. [1988], Chapter 12
"""
import numpy as np 
import pandas as pd
import scipy.optimize as opt

judge = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/judge.txt",names=['L','K','Q'],sep='\s+')
L, K, Q = judge.L, judge.K, judge.Q

def ces(b):
    e=np.log(Q)-(b[0]+b[3]*np.log(b[1]*L**b[2]+(1-b[1])*K**b[2]))
    return e

def sse_ces(b):
    e=ces(b)
    return sum(e**2)

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
    
b0=[1,0.5,-1,-1]
res0=opt.minimize(sse_ces,b0,method='Nelder-Mead')
res=opt.minimize(sse_ces,res0.x,method='bfgs')
res

# computing var-cov matrix
from numpy.linalg import inv
import statsmodels.tools.numdiff as nd
# gauss-newton approximation of the hessian matrix
gv = nd.approx_fprime(res.x,ces)
H = (gv.T @ gv) 
# assume homoscedasticity
v = sse_ces(res.x)/len(judge)
var= v * inv(H)
# alternatively, try ...
# var = v * res.hess_inv
print_output(res.x,var,['b1','b2','b3','b4'])


# alternatively, a better approach is
# Using lmfit package which depends on leastsq and least_squares
import lmfit as nlm

# data variables are Q, L, K
def ces(params):
    "CES Production Function"
    b0=params['b0']
    b1=params['b1']
    b2=params['b2']
    b3=params['b3']
    e = np.log(Q)-(b0+b3*np.log(b1*L**b2+(1-b1)*K**b2))
    return e

b = nlm.Parameters()
# b.add_many(('b0', 1.0), ('b1', 0.5), ('b2', -1.0), ('b3', -1.0))
b.add('b0',value=1.0)
b.add('b1',value=0.5,min=1.0e-6,max=1.0)
b.add('b2',value=-1.0)
b.add('b3',value=-1.0)
b
# using default Levenberg-Marquardt method (leastsq)
out = nlm.minimize(ces,b) 
nlm.report_fit(out)

out1 = nlm.minimize(ces,b,method='least_squares')
nlm.report_fit(out1)

# need to install numdifftools for some methods
out2 = nlm.minimize(ces,b,method='bfgs')
nlm.report_fit(out2)

# with parameter restriction b3=1/b2
b.add('b3',expr='1/b2')
b
out3 = nlm.minimize(ces,b,method='bfgs')
nlm.report_fit(out3)

# minimize the sum-of-squares of errors directly    
def sse_ces(params): 
    e = ces(params)
    return(sum(e**2))

res = nlm.minimize(sse_ces,b,method='bfgs') 
nlm.report_fit(res)



