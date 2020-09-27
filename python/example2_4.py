"""
 Example 2.4
 Data Analysis
"""
%cd C:/Course20/ceR/python

import numpy as np
import pandas as pd
data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)

y=data['EM']
x=data.drop(['EM'],axis=1)
one=pd.Series(np.ones(16))
x=pd.concat([one,x],join='inner',axis=1)

from numpy.linalg import inv, eig, cond
xx = x.T @ x
xy = x.T @ y
invxx = inv(xx)
b = invxx @ xy
b

cm = eig(xx)
cn = np.sqrt(np.max(cm[0])/np.min(cm[0]))
cn
# compare with built-in cond
cond(x)
