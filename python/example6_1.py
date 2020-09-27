"""
# Example 6.1
# One-Variable Scalar-Valued Function
# References: based on STA-663-2017 1.0 Documentation
# http://people.duke.edu/~ccc14/sta-663-2017/14C_Optimization_In_Python.html
# https://docs.scipy.org/doc/scipy/reference/optimize.html
"""

import numpy as np 
import scipy.optimize as opt # from scipy import optimize
import matplotlib.pyplot as plt

# def f(x): return np.log(x)-x**2
# using one-line function
f = lambda x: np.log(x)-x**2
v=0.5
f(v)

# negative of f(x) and its derivative
# def nf(x): return -1*f(x)
# def nf1(x): return 2*x - 1/x
nf = lambda x: -1*f(x)
nf1 = lambda x: 2*x - 1/x
nf2 = lambda x: 2 + 1/(x**2)
nf(v)
nf1(v)
nf2(v)

# using numeric derivatives
# pip install numdifftools
import numdifftools as nd
nd.Gradient(nf,method='complex')(v)
nd.Hessian(nf,method='complex')(v)

# res1 = opt.minimize_scalar(nf)  # method='brent'
res1 = opt.minimize(nf,v,method='bfgs')
res1.fun
res1.x
res1.jac
res1.hess_inv
res2 = opt.minimize(nf,v,jac=nf1,method='newton-cg')
res2

x = np.linspace(1.0e-6,4.0,100)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.scatter(res2.x,-res2.fun)
plt.plot(x,f(x))
plt.show()

fig, (ax1,ax2) = plt.subplots(2,sharex=True)
ax1.set_ylabel('f',rotation=0)
ax2.set_ylabel('df',rotation=0)
ax2.set_ylim(-10,10)
ax1.plot(x,f(x))
ax2.plot(x,-nf1(x))
ax1.scatter(res2.x,-res2.fun)
ax1.vlines(res2.x,ymin=-15,ymax=-res2.fun,linestyles='dotted')
ax2.vlines(res2.x,ymin=-10,ymax=10,linestyles='dotted')
ax2.hlines(0,xmin=0,xmax=4,linestyles='dotted')
plt.show()

