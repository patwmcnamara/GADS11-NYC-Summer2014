#<<<<<<< HEAD
import csv
import numpy as np
import pandas as pd
from dateutil import parser
import pylab as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import scale
from numpy import inf
import scipy.stats as stats
import pylab
#=======

### IMPORT DATA ###

batting_salary = pd.read_csv("/Users/patrickmcnamara/Documents/GA_DataScience/Teaching/Summer14/GADS11-NYC-Summer2014/projects/Project2/baseball.csv")
pitching = pd.read_csv("/Users/patrickmcnamara/Documents/GA_DataScience/Teaching/Summer14/GADS11-NYC-Summer2014/projects/Project2/pitching.csv")

# DROP UNWANTED VARIABLES #
batting_salary = batting_salary.drop(['lahmanID', 'managerID', 'birthYear', 'birthMonth', 'birthDay', 
	'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry', 
	'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick','bats', 
	'throws', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID', 
	'bbrefID', 'deathDate', 'birthDate','teamID', 'lgID', 'stint','G_batting','X2B', 'X3B',
	'CS', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'G_old', 'hofID'], axis =1)

# KEEP VARIABLES IN PITCHING DATA THAT AREN'T IN BATTING DATA #
keep_cols = list(set(pitching.columns)-set(batting_salary.columns))
keep_cols = keep_cols + ['playerID','yearID']
pitching = pitching[keep_cols]
pitching = pitching.drop(['GIDP','SH','SF'], axis=1)
pitching = pitching[['ERA','SO','playerID','yearID']]

# MERGE DATASETS #
data = pd.merge(batting_salary, pitching, on=['playerID','yearID'], how='outer')
data = data.drop(['playerID','yearID'], axis=1)

# DROP PITCHERS FROM DATASET #
index = data['ERA'].index[data['ERA'].apply(np.isnan)]
slimdata = data.loc[index]
slimdata = slimdata.drop(['ERA','SO'], axis=1)

### CHECKING VARIABLE RELATIONSHIPS ###
# SHRINKING THE DATA TO MAKE VISUALIZATIONS EASIER #

slimdata['random'] = np.random.randn(len(slimdata))
slimdata = slimdata[slimdata.random > 1]
del slimdata['random']

# COLLINEARITY HISTOGRAM #
pd.tools.plotting.scatter_matrix(slimdata, alpha=0.2, diagonal='hist')
plt.show()

# LOG TRANSFORMATIONS WHERE NECESSARY #
slimdata.SB = np.log(slimdata.SB)
slimdata.HR = np.log(slimdata.HR)
slimdata.BB = np.log(slimdata.BB)
slimdata.RBI = np.log(slimdata.RBI)
slimdata.salary = np.log(slimdata.salary)

# REPLACE INF VALUES WITH NAN #
slimdata = slimdata.replace([inf, -inf], np.nan)

# DROP NAN #
slimdata = slimdata.dropna()

### PLOTTING SCATTERPLOT MATRIX FOR COLLINEARITY ###

# HISTOGRAM #
pd.tools.plotting.scatter_matrix(slimdata, alpha=0.2, diagonal='hist')
plt.show()

# KERNEL DENSITY #
pd.tools.plotting.scatter_matrix(slimdata, alpha=0.2, diagonal='kde')
plt.show()

### RUNNING REGRESSION MODEL ###

# CREATING INTERCEPT #
slimdata['intercept'] = 1

# DEFINING IVs & DVs #
X = slimdata.drop(['salary'], axis = 1)
y = slimdata['salary']

# RUNNING REGRESSION #

model = sm.OLS(y, X)
results = model.fit()
results.summary()

### NORMALIZATION ###

# BOX PLOT FOR OUTLIERS #

slimdata.boxplot()
plt.show()

# SCALING # Mean-center then divide by std dev

data_norm = pd.DataFrame(scale(slimdata), index=slimdata.index, columns=slimdata.columns)
data_norm.boxplot()
plt.show()

### RUNNING REGRESSION MODEL AGAIN ###
data_norm['intercept'] = 1
X = data_norm.drop(['salary'], axis = 1)
y = data_norm['salary']

model2 = sm.OLS(y, X)
results2 = model2.fit()
results2.summary()

### INFLUENCE PLOT FOR SINGLE OBSERVATIONS ###

fig, ax = plt.subplots(figsize=(10,10))
fig = sm.graphics.influence_plot(results2, ax=ax, criterion="cooks")
plt.show()

# INFLUENCE TABLE #
influence = results2.get_influence()
influence.summary_frame()['cooks_d'].order()

# THE EFFECT OF RESHAPING/DROPPING VARIABLES #
res_dropped = results.params / results2.params * 100 - 100
'''Create new regressions and see what these look like
after dropping extremely influential variables'''

### RESIDUALS PLOT ###
plt.scatter(results2.norm_resid(), results2.fittedvalues)
plt.xlabel('Fitted Values')
plt.ylabel('Normalized residuals')
plt.show()
'''Here we're looking for something resembling a shotgun blast.
Random points with no identifiable structure'''

### LOOKING AT INDIVIDUAL VARIABLES ###

# PARTIAL REGRESSION PLOTS #
fig = plt.figure(figsize=(10,10))
fig = sm.graphics.plot_partregress_grid(results2, fig=fig)
plt.show()
'''Here we want to see a linear relationship'''

fig = plt.figure(figsize=(10,10))
fig = sm.graphics.plot_regress_exog(results2, 'H', fig=fig)
plt.show()
'''2x2 plot containing DV and fitted values with CIs vs. selected IV,
residuals vs. the IV, a partial regression plot, and a CCPR plot.
Don't worry about the CCPR plot'''
