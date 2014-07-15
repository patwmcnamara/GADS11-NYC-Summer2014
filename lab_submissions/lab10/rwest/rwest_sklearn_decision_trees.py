# Robert West
# LAB 10
import pandas
import numpy as np

############################################################
### Read in data and clean using pandas ###
############################################################

# read in data using pandas
titanic_data = pandas.io.parsers.read_csv('data/titanic.csv')

# We keep the class, age and sex variables
titanic_data = titanic_data[['survived','pclass','sex','age']]

# We have missing values for age, so we're going to assign the mean value
titanic_data['age'][titanic_data['age'].isnull()] = np.mean(titanic_data['age'])

### Encode sex as a categorical variable 
sex = pandas.get_dummies(titanic_data['sex'])
titanic_data = titanic_data.join(sex)

### Now we encode 'class', which has more than 2 possible values - pandas makes this easy
pclass = pandas.get_dummies(titanic_data['pclass'])
pclass.rename( columns = {1: 'first_class', 2: 'second_class', 3: 'third_class'}, inplace = True)
titanic_data = titanic_data.join(pclass)

titanic_data.drop(['sex','pclass','female'],axis=1,inplace=True)

# Inspect
print titanic_data.head()

# Create training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_data[['age','male','first_class','second_class','third_class']], titanic_data['survived'], test_size=0.25, random_state=33)

############################################################
### DECISION TREES ###
############################################################

# Fit a decision tree with the data using entropy to measure information gain
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)
clf = clf.fit(X_train,y_train)

# Show the built tree, using pydot
import pydot,StringIO
dot_data = StringIO.StringIO() 

tree.export_graphviz(clf, out_file=dot_data, feature_names=['age','male','first_class','second_class','third_class']) 

dot_data.getvalue()

pydot.graph_from_dot_data(dot_data.getvalue())

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf('titanic.pdf')
print '\nimage created!'

# Create function to measure model performance
from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred=clf.predict(X)   
    if show_accuracy:
        print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n"
    if show_classification_report:
        print "Classification report"
        print metrics.classification_report(y,y_pred),"\n"
    if show_confusion_matrix:
        print "Confusion matrix"
        print metrics.confusion_matrix(y,y_pred),"\n"

# Measure Accuracy, precision, recall, f1 in the training set
# Precision = true positives/(true positives + false positives). The ability of the classifier to not label a negative sample as positive
# Recall = true positives/(true positives + false negatives). The ability of the classifier to find all positive samples
# f1  = 2 * (precision * recall) / (precision + recall)
measure_performance(X_train,y_train,clf, show_classification_report=True, show_confusion_matrix=True)

# Perform leave-one-out cross validation to better measure performance, reducing variance
from sklearn.cross_validation import cross_val_score, LeaveOneOut
from scipy.stats import sem

# Inspect documentation for LeaveOneOut
# help(LeaveOneOut)

def loo_cv(X_train,y_train,clf):
    # Perform Leave-One-Out cross validation
    loo = LeaveOneOut(X_train[:].shape[0])
    scores=np.zeros(X_train[:].shape[0])
    for train_index,test_index in loo:
        X_train_cv, X_test_cv= X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv= y_train[train_index], y_train[test_index]
        clf = clf.fit(X_train_cv,y_train_cv)
        y_pred=clf.predict(X_test_cv)
        scores[test_index]=metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

loo_cv(X_train, y_train,clf)

# Try to improve performance using Random Forests
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10,random_state=33)
clf = clf.fit(X_train,y_train)
loo_cv(X_train,y_train,clf)

# Attempt 1
clf_dt=tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)
clf_dt.fit(X_train,y_train)
measure_performance(X_test,y_test,clf_dt)

# Inspect documentation for DecisionTreeClassifier
# help(tree.DecisionTreeClassifier)

# Attempt 2
clf_dt=tree.DecisionTreeClassifier(criterion='gini', max_depth=3,min_samples_leaf=10)
clf_dt.fit(X_train,y_train)
measure_performance(X_test,y_test,clf_dt)

### A New Measure: the ROC and Area Under a Curve (AUC)

# One way we can score a binary classification is by plotting the reciever 
# operating characteristic and determining the value of the area under curve (AUC). 
# Like above, our goal is to see an AUC as close to 1 as possible.

# Syntax for roc_curve is roc_curve(actual, prediction, [pos_label if it's not 1])
predictions = [p[1] for p in clf_dt.predict_proba(X_train)]
fpr_p, tpr_p, thresholds_p = metrics.roc_curve(y_train,predictions)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_figwidth(10)
fig.suptitle('AUC for Decision Tree Classifier Predicting Titanic Survivors')

ax1 = plt.subplot(1, 2, 1)
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
ax1.plot(fpr_p, tpr_p)

fpr, tpr, thresholds = metrics.roc_curve(y_train,clf_dt.predict(X_train))
ax2 = plt.subplot(1, 2, 2)
ax2.set_xlabel('false positive rate')
ax2.set_ylabel('true positive rate')
ax2.plot(fpr, tpr)

print "False-positive rate:", fpr
print "True-positive rate: ", tpr
print "Thresholds:         ", thresholds

fig.show()

metrics.roc_auc_score(y_train, predictions)
metrics.roc_auc_score(y_train,clf_dt.predict(X_train))

'''
HOMEWORK
Change some of the assumptions we've made throughout the lab to see how that changes the accuracy; Imputation, tree depth, samples, etc.
Try to find the most accurate model you can; talk about what you did, address the bias-variance tradeoff.
How could your accuracy be improved? Think internally to our model building and externally as well.
'''