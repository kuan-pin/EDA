# Example 6.2
# Two-Variable Scalar-Valued Function
# References: based on STA-663-2017 1.0 Documentation
# http://people.duke.edu/~ccc14/sta-663-2017/14C_Optimization_In_Python.html
# https://docs.scipy.org/doc/scipy/reference/optimize.html

import numpy as np 
import scipy.optimize as opt # from scipy import optimize

# gx = lambda x: (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
def g(x): return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

#1st derivative of gx 
def g1(x): 
    gn1 = 4*x[0]*(x[0]**2+x[1]-11) + 2*(x[0]+x[1]**2-7) 
    gn2 = 2*(x[0]**2+x[1]-11) + 4*x[1]*(x[0]+x[1]**2-7) 
    return np.array([gn1, gn2])

#2nd derivative of gx 
def g2(x): 
    gn11 = 12*x[0] + 4*x[1] -42 
    gn22 = 4*x[0] + 12*x[1]**2 -26 
    gn12 = 4*(x[0] + x[1]) 
    return np.array([[gn11, gn12], [gn12, gn22]])

v = [1.0,1.0]
g(v)
g1(v)
g2(v)

# using numeric derivatives
import numdifftools as nd
nd.Gradient(g)(v)
nd.Hessian(g)(v)

# find 4 minima of f(x,y) at: f1=0 0, f2= positive definite
# (3,2), (3.5844,-1.8481), (-3.7793,-3.2832), (-2.8051,3.1313)
# there is 1 maximum: (-0.27084,-0.92304) with f=181.62
# saddle points: (0.08668,2.88430), (3.38520,0.07358), (-3.07300,-0.08135)

res1a = opt.minimize(g,v)  # method=’Nelder-Mead’
res1a
res1b = opt.minimize(g,v,method='bfgs')
res1b
res1c = opt.minimize(g,v,method='powell')
res1c
res1d = opt.minimize(g,v,method='newton-cg',jac=g1)
res1d

v = [-2, 3]
res2a = opt.minimize(g,v)  # method=’Nelder-Mead’
res2a
res2b = opt.minimize(g,v,method='bfgs')
res2b
res2c = opt.minimize(g,v,method='powell')
res2c
res2d = opt.minimize(g,v,method='newton-cg',jac=g1)
res2d

v = [2, -3]
res3a = opt.minimize(g,v)  # method=’Nelder-Mead’
res3a
res3b = opt.minimize(g,v,method='bfgs')
res3b
res3c = opt.minimize(g,v,method='powell')
res3c
res3d = opt.minimize(g,v,method='newton-cg',jac=g1)
res3d

v = [-2, -3]
res4a = opt.minimize(g,v)  # method=’Nelder-Mead’
res4a
res4b = opt.minimize(g,v,method='bfgs')
res4b
res4c = opt.minimize(g,v,method='powell')
res4c
res4d = opt.minimize(g,v,method='newton-cg',jac=g1)
res4d

# Using graph to find the minima
# 3D surface plot 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

fig = plt.figure(figsize=(8,6)) 
ax3d = Axes3D(fig)

x1 = np.linspace(-5, 5, 500)
x2 = np.linspace(-5, 5, 500)
x = np.meshgrid(x1, x2)

ax3d.plot_surface(x[0],x[1],g(x),rstride=10,cstride=10,cmap='jet')
# rstride&cstride are steps, which can be adjusted
ax3d.set_xlabel('x1', fontsize=12)
ax3d.set_ylabel('x2', fontsize=12)
ax3d.set_zlabel('y', fontsize=12)
plt.show()

# 2D contour plot
C = plt.contour(x[0],x[1],g(x),10)
plt.clabel(C, inline=True, fontsize=9)
plt.show()

plt.figure(figsize=(8,6))
C = plt.contour(x[0],x[1],g(x),100)
plt.clabel(C, inline=True, fontsize=8)
plt.scatter(res1d['x'][0],res1d['x'][1])
plt.scatter(res2d['x'][0],res2d['x'][1])
plt.scatter(res3d['x'][0],res3d['x'][1])
plt.scatter(res4d['x'][0],res4d['x'][1])
plt.show()
