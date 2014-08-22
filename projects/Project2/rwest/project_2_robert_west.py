
import pandas
import numpy 
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale

numpy.random.seed(seed=1)

# read in data from csv - and make the index the year ID so that we can create time series

df = pandas.io.parsers.read_csv('baseball.csv',parse_dates=True)
df = df.sort(['playerID','yearID'])
df.set_index('yearID',inplace=True)
 
# Slugging Average [SA or SLG] (with no denominator!)
# SA = Number of (Singles + [2 x Doubles] +[ 3 x Triples] + [4 x Home Runs]) divided by At Bats
df['SA'] = df.R +  2*df.X2B + 3*df.X3B+ 4*df.HR
 
new_df = pandas.DataFrame()
for player in set(df['playerID']):
    temp_data = df[df['playerID']==player]
    temp_data['cum_SA'] = temp_data['SA'].cumsum()
    new_df = new_df.append(temp_data)

# SHRINKING THE DATA TO MAKE VISUALIZATIONS EASIER #
new_df['random'] = numpy.random.randn(len(new_df))
new_df = new_df[new_df.random > 1]
del new_df['random']


simple_data = new_df[['salary','cum_SA','playerID','weight','height','bats','hofID']]

### Encode bats as a categorical variable 
bats = pandas.get_dummies(simple_data['bats'])
#bats.rename( columns = {'B': 'bats_both_dummy', 'L': 'bats_left_dummy', 'R': 'bats_right_dummy'}, inplace = True)
#simple_data = simple_data.join(bats)
#simple_data = simple_data.drop(['bats'])

### Encode 'hall_of_fame' as a categorical variable
#hofame = pandas.get_dummies(~simple_data['hofID'].isnull())
#hofame.rename( columns = {'True': 'hoffame_dummy'}, inplace = True)
#simple_data = simple_data.join(hofame)
simple_data = simple_data.drop(['hofID'])

simple_data['log_salary'] = numpy.log(simple_data['salary'])

# REPLACE INF VALUES WITH NAN #
simple_data = simple_data.replace([numpy.inf, -numpy.inf], numpy.nan)

# remove nans
simple_data = simple_data.dropna()

# get yearID back from index
simple_data['yearID'] = simple_data.index

# remove inflation by subtracting the annual mean
#grouped = simple_data.groupby(level=0)
#yearly_avg_salary = grouped.mean().salary

#simple_data = simple_data.join(pandas.DataFrame({'avg_salary':yearly_avg_salary}),how='left')
#simple_data['adj_salary'] = simple_data['salary'] - simple_data['avg_salary']


### NORMALIZATION ###
# SCALING # Mean-center then divide by std dev
# simple_data = simple_data[['yearID','cum_SA', 'height','log_salary']]
# simple_data = pandas.DataFrame(scale(simple_data), index=simple_data.index, columns=simple_data.columns)

# training data
simple_data['random'] = numpy.random.rand(len(simple_data))
train_df = simple_data[simple_data.random > 0.4]
train_df = simple_data

train_df['intercept'] = 1
#X = train_df[['bats_both_dummy','bats_left_dummy',  'bats_right_dummy','yearID','height','cum_SA','intercept']] 
X = train_df[['cum_SA','intercept']] 
#X = train_df[['yearID','height','cum_SA','intercept']] 
y = train_df['log_salary']
model = sm.OLS(y, X)
results = model.fit()
print results.summary()

test = simple_data[simple_data.random <= 0.4]
#test_X = train_df[['bats_both_dummy','bats_left_dummy',  'bats_right_dummy','yearID','height','cum_SA','intercept']] 
#test_X = test[['yearID','height','cum_SA','intercept']] 
test_X = test[['cum_SA','intercept']] 
test_y = test['salary']
y_hat = numpy.exp(results.predict(test_X))

avg_abs_err = numpy.mean(numpy.abs(test_y - y_hat))
print avg_abs_err