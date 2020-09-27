# Example 8.1 Probit Model of Economic Education
# Binomial Regression Model: Probit and Logit
# Probit Model using statsmodels package

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf 

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/grade.txt',sep='\s+', nrows=32) 
data = data.dropna()
print(data.describe())

X = data[['GPA','TUCE','PSI']]
y = data['GRADE']
model1 = sm.Probit(y,sm.add_constant(X)).fit()
model1.summary()

model2 = sm.Probit.from_formula('GRADE ~ GPA + TUCE + PSI',data=data).fit()
model2.summary()

# same as model2
model3 = smf.probit('GRADE ~ GPA + TUCE + PSI', data=data).fit()
model3.summary()

# using GLM
model4 = smf.glm(formula='GRADE ~ GPA + TUCE + PSI', 
                family=sm.families.Binomial(link = sm.genmod.families.links.probit),
                data=data).fit()
model4.summary()

# probit model interpretation
model2.params
# predicted probability of each obs
model2.predict(data)
# marginal effects at the mean of each regressor
margeff = model2.get_margeff('mean')
margeff.summary()

