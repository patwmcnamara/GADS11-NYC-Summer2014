# do these first:
# 1) brew install graphviz
# 2) pip install pydot
# 3) set directory to lesson 10
import csv
import numpy as np
with open('/Users/patrickmcnamara/Documents/GA_DataScience/Teaching/Summer14/GADS11-NYC-Summer2014/lessons/lesson10_decision_trees_random_forests/data/titanic.csv', 'r') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
    # Header contains feature names
    row = titanic_reader.next()
    feature_names = np.array(row)
    
    # Load dataset, and target classes
    titanic_X, titanic_y = [], []
    for row in titanic_reader:  
        titanic_X.append(row)
        titanic_y.append(row[0]) # The target value is "survived"
    # Changing to arrays
    titanic_X = np.array(titanic_X)
    titanic_y = np.array(titanic_y)


# Inspecting last row, header, features and target
print row, feature_names, titanic_X[0], titanic_y[0]

# We keep the class, age and sex variables
titanic_X = titanic_X[:, [1, 4, 3]]
feature_names = feature_names[[1, 4, 3]]

# We have missing values for age, so we're going to assign the mean value
ages = titanic_X[:, 1]
mean_age = np.mean(titanic_X[ages != '', 1].astype(np.float))
titanic_X[titanic_X[:, 1] == '', 1] = mean_age

# Encode sex as a categorical variable 
from sklearn.preprocessing import LabelEncoder
# This will normalize our class variables by giving them easily interpreted labels
enc = LabelEncoder()
# Creating categorical classes for sex
label_encoder = enc.fit(titanic_X[:, 2])
print "Categorical classes:", label_encoder.classes_

# Creating numerical classes for sex
integer_classes = label_encoder.transform(label_encoder.classes_)
print "Integer classes:", integer_classes
t = label_encoder.transform(titanic_X[:, 2])
titanic_X[:, 2] = t

# Inspect
print feature_names
print titanic_X[5], titanic_y[5]

# Now we encode 'class', which has more than 2 possible values
from sklearn.preprocessing import OneHotEncoder
enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:, 0])
print "Categorical classes:", label_encoder.classes_
integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3, 1)
print "Integer classes:", integer_classes
enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)

# First, convert classes to 0-(N-1) integers using label_encoder
num_of_rows = titanic_X.shape[0]
t = label_encoder.transform(titanic_X[:, 0]).reshape(num_of_rows, 1)
# Second, create a sparse matrix with three columns, each one indicating if the instance belongs to the class
new_features = one_hot_encoder.transform(t)
# Add the new features to titanix_X
titanic_X = np.concatenate([titanic_X, new_features.toarray()], axis = 1)
#Delete converted column
titanic_X = np.delete(titanic_X, [0], 1)
# Update feature names
feature_names = ['age', 'sex', 'first_class', 'second_class', 'third_class']
# Convert to numerical values
titanic_X = titanic_X.astype(float)
titanic_y = titanic_y.astype(float)

# Inspect
print feature_names
print titanic_X[0], titanic_y[0]

# Create training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size=0.25, random_state=33)

### DECISION TREES ###

# Fit a decision tree with the data using entropy to measure information gain
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)
clf = clf.fit(X_train,y_train)

# Show the built tree, using pydot
import pydot,StringIO
dot_data = StringIO.StringIO() 

tree.export_graphviz(clf, out_file=dot_data, feature_names=['age','sex','1st_class','2nd_class','3rd_class']) 

dot_data.getvalue()

pydot.graph_from_dot_data(dot_data.getvalue())

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf('titanic.pdf')
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
help(LeaveOneOut)

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
help(tree.DecisionTreeClassifier)

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
''''