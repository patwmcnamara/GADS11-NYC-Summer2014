import numpy
import pandas
import sklearn
import csv

#Test the import functionality against pandas.DataFrame.from_csv

def open_titanic(filepath):
    with open(filepath, 'r') as csvfile:
        titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
    # Header contains feature names
        row = titanic_reader.next()
        feature_names = numpy.array(row)
    
    # Load dataset, and target classes
        titanic_X, titanic_y = [], []

        for row in titanic_reader:  
            titanic_X.append(row)
            titanic_y.append(row[0]) # The target value is "survived"
    # Changing to arrays
    titanic_X = numpy.array(titanic_X)
    titanic_y = numpy.array(titanic_y)

    return feature_names, titanic_X, titanic_y

def bens_fun(filepath):
    #A gentle suggestion for simplifying the data loading & cleaning...
    data = pandas.DataFrame.from_csv('./titanic.csv', index_col = None)

    #define our X variates, and make a copy instead of operating on the data itself
    xs = data[['pclass', 'age', 'sex']].copy()

    #encode sex into binary classification
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    xs['sex'] = le.fit_transform(xs['sex'])

    #pandas.get_dummies() instead of OneHotEncoder() maybe?
    pclass = pandas.get_dummies(xs['pclass'])
    pclass.rename( columns = {1: '1st_class', 2: '2nd_class', 3: '3rd_class'}, 
                   inplace = True)
    xs = xs.join(pclass)
    xs = xs[['age', 'sex', '1st_class', '2nd_class', '3rd_class']].copy()

    return xs

