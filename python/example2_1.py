"""
 Example 2.1
 get/set working directory
"""
# using magic functions
%pwd
%cd C:/Course20/ceR/python
%pwd
%ls
%env

# get help
# interactive help utility
help()
help('array')

# install and use packages 
# pip install numpy
# numpy, scipy, matplotlib, pandas, statsmodels, ...
import numpy as np
np.array?

# Let's Begin
A = np.array([[1,2,3],[0,1,4],[0,0,1]])
dir(A)  # see the details of an object A
type(A)
np.ndim(A)   # A.ndim
np.shape(A)  # A.shape
A
np.transpose(A)  # A.transpose(), A.T
np.trace(A)      # A.trace()

# part of array
A[0,1]
A[1,:]
A[1,1:]
A[-1,:]
A[1:,-1]

B = np.array([2,7,1])
np.ndim(B)
np.shape(B)
B
B.reshape(-1,1)  # B.reshape(3,1)
B.reshape(1,-1)  # B.reshape(1,3)

C = np.array([[2,7,1]])
np.ndim(C)
np.shape(C)
C
C.T

C1 = np.full((3,3),C)
C1
C2 = np.full((3,3),C.T)
C2

# inner product 
np.dot(B,B)    # B @ B
np.dot(C,C.T)  # C @ C.T
np.dot(C.T,C)  # C.T @ C

# matrix multiplication using @
A @ C.T   # np.dot(A,C.T)
# element by element mulplication using *
A * C     # np.multiply(A,C)
A * C.T

%who   
# %who_ls, %whos
%whos
# clean up (no prompt)
%reset -f
