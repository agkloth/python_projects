# Importing packages

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF 
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize , poly)

# Simple Linear Regression

Boston = load_data("Boston") 
print(Boston.columns)

X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),'lstat': Boston['lstat']})
print(X[:4])
# creating new DataFrame using pandas library
# intercept: creating column called 'intercept'; assigns an array of ones 
# (using np.ones()) with the same number of rows as Boston DataFrame
# purpose of adding this column of ones is to act as the intercept in regression model.
# Boston.shape[0] gives number of rows in the Boston dataset

y=Boston['medv']
model = sm.OLS(y,X)
results=model.fit()
#extracting response variable medv
#sm.OLS specifies model
#model.fit() does actual fitting

summarize(results)

# Prediction and Confidence Intervals

new_df = pd.DataFrame({'lstat':[5, 10, 15]}) 
design = MS(['lstat'])
X = design.fit_transform(Boston)
newX = design.transform(new_df)
print(newX)
# create a new data frame for variable lstat with values for this 
# variable we want to make predictions
# MS stands for ModelSpec, function from ISLP model packages
# argument ['lstat'] means that specifying 'lstat' as predictor variable in design matrix
# design is an object of ModelSpec class; contains info on which columns to be
# used
# fit_transform() method looks at data to see which columns are present and does transformations
# now design object fitted with boston data
# and transform () method creates correspoding corresponding matrix model

new_predictions = results.get_prediction(newX)
new_predictions.predicted_mean
# compute predictions at newX and view them by using predicted_mean attribute

new_predictions.conf_int(alpha=0.05)
# producing confidence intervals for predicted values

new_predictions.conf_int(obs=True, alpha=0.05)
# producing prediction intervals for predicted values

# Multiple Linear Regression

X = MS(['lstat', 'age']).fit_transform(Boston) 
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)
#use ModelSpec() transform to construct required model matrix and response

terms = Boston.columns.drop('medv')
terms
# short-cut way to perform regression using all predictors; instead, taking out
# response variable

X=MS(terms).fit_transform(Boston)
model2 = sm.OLS(y,X)
results2 = model2.fit()
summarize(results2)

# Qualitative Predictors

Carseats = load_data('Carseats')
print(Carseats.columns)
# predictor ShelveLoc takes on three possible values, Bad, Medium, and Good
# given qualitative variable ModelSpec() generates dummy variables automatically
# their columns sum to one, so to avoid collinearity with an intercept, 
# the first column is dropped

allvars = list(Carseats.columns.drop('Sales')) 
y = Carseats['Sales']
final = allvars + [('Income', 'Advertising'),
('Price', 'Age')]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X) 
summarize(model.fit())
# removing Sales as it is our response variable we trying to predict
# turning column into python list
# assigning Sales column as response variable
# concatenating allvars and two tuples
# two tuples correspond to two interaction variables
# .fit() method estimates coefficients for each predictor variable
# by minimizing the error between the predicted and actual values of response 
# variable 
# in the output, you will see column ShelveLoc[Bad] dropped, since it is the 
# first level of ShelveLoc
