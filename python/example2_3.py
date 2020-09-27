"""
 Example 2.3
 Data Transformation
"""
%cd C:/Course20/ceR/python

import numpy as np
import pandas as pd
# data=pd.read_table('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)
data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)

PGNP=np.log(data['PGNP'])
GNP=np.log(data['GNP']/1000)   
POPU=np.log(data['POP']/1000)
EM=np.log(data['EM']/1000)

x=np.array([PGNP,GNP,POPU,EM])
x.T
