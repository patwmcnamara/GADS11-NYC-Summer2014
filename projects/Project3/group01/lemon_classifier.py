"""
Define the Problem: Car is a good buy (0) or bad buy (1)
Type of Problem: Classification problem

"""

# IMPORT MODULES
import pandas as pd
from sklearn import tree

# IMPORT DATA
train = pd.read_csv("/Users/patrickmcnamara/Documents/GA_DataScience/Teaching/Summer14/GADS11-NYC-Summer2014/projects/Project3/lemon_training.csv")
test = pd.read_csv("/Users/patrickmcnamara/Documents/GA_DataScience/Teaching/Summer14/GADS11-NYC-Summer2014/projects/Project3/lemon_test.csv")

# CLEAN DATA
features = list(train.columns)
features.remove('IsBadBuy') '''Target variable'''
'''Remove & Create other features here'''

# CREATE TRAINING & TEST DATA
training_X = train[features].values
training_y = train['IsBadBuy'].values

# CLASSIFY WITH A ML ALGORITHM






# SCORE IT






# EXPORT PREDICTIONS #
'''Export a CSV of your predictions for each car in the TEST data'''