from __future__ import print_function
import pandas
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import fb_sales_library
import pdb

df = pandas.DataFrame.from_csv('Organic Sales.csv')

# all data line plot

f, ax = plt.subplots() 
p1 = ax.plot(df.index,df['Paid Sales'], lw=2,color='#D1B9D4')
p2 = ax.plot(df.index,df['Organic Sales'],lw=2, color='#D1D171')
ax.set_title('Untracked Sales, Paid Sales and Spend')
ax.set_ylabel('Sales')
plt.setp(ax.get_xticklabels(), rotation=30, fontsize=6)
ax2 =ax.twinx()
p3 = ax2.plot(df.index, df['Spent'],lw=2, color='#84DEBD' ,linewidth=1)
ax2.set_ylabel('Spend')
ax2.legend( (p1[0],p2[0],p3[0]), ('Paid Sales','Untracked Sales','Spend'),prop={'size':8})   
plt.savefig('Paid Sales_Untracked Sales_spend.jpg')
plt.show()
plt.close()

# All Data scatter plot
X = df['Paid Sales'].reset_index().values[:,1]
y = df['Organic Sales']
fig = plt.subplots(figsize=(8,6))
plt.scatter(X,y,facecolor='#D1B9D4',edgecolor='black')
plt.xlabel('Paid Sales')
plt.ylabel('Untracked Sales')
plt.title('Paid Sales VS Untracked Sales')
plt.savefig('Paid Sales VS Untracked Sales.jpg')
plt.show()
plt.close()

# All Data scatter plot
X = df['Paid Sales'].reset_index().values[:,1]
y = df['Organic Sales']
fig = plt.subplots(figsize=(8,6))
plt.scatter(X,y,c=y.index.to_datetime())
cb = plt.colorbar()
n = len(y.index)
plt.xlabel('Paid Sales')
plt.ylabel('Untracked Sales')
plt.title('Paid Sales VS Untracked Sales - with time heatmap')
#pdb.set_trace()
ticks = numpy.array([str(y.index[17])[:10], str(y.index[35])[:10], str(y.index[53])[:10], str(y.index[71])[:10], str(y.index[89])[:10],str(y.index[107])[:10],str(y.index[125])[:10]])
cb.ax.set_yticklabels(ticks)
plt.savefig('Paid Sales VS Untracked Sales - with time heatmap.jpg')
plt.show()
plt.close()
