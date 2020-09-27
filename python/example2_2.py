"""
 Example 2.2
 File I/O
"""
%cd C:/Course20/ceR/python
# install package when first use
# pip install pandas
import numpy as np
import pandas as pd
# data=pd.read_table('http://web.pdx.edu/~crkl/ceR/data/longley.txt',nrows=16)
data=pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/longley.txt',sep='\s+',nrows=16)

data.head()
data.columns
data.values

PGNP=data['PGNP']
GNP=data['GNP']/1000   
POPU=data['POP']/1000
EM=data['EM']/1000

x=np.array([PGNP,GNP,POPU,EM])
x.T   # x.transpose()

